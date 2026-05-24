"""
LLMClient — unified LLM interface for discovery-behavioral-analysis.

Supports providers: anthropic, openai, openai_compatible, google_vertex.
Reads model config from configs/models.yaml.
Reads API keys from environment: ANTHROPIC_API_KEY, OPENAI_API_KEY.
For google_vertex, uses a service-account JSON whose path is resolved from
config (`credentials_path` / `credentials_path_env`) or
$GOOGLE_APPLICATION_CREDENTIALS as a final fallback.

Usage:
    from src.llm_client import LLMClient
    client = LLMClient("claude-opus-4-6")
    result = client.complete("Write a circle packing algorithm.")
    print(result["text"])
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import yaml


class LLMClient:
    """
    Thin wrapper around anthropic / openai SDKs with exponential backoff.

    Args:
        model_name:         Key matching the `name` field in configs/models.yaml.
        models_config_path: Path to the models YAML config (default: configs/models.yaml).
    """

    def __init__(
        self,
        model_name: str,
        models_config_path: str = "configs/models.yaml",
    ) -> None:
        self.model_name = model_name
        config = self._load_model_config(model_name, models_config_path)
        self.provider: str = config["provider"]
        self.model_id: str = config["model_id"]
        self.base_url: Optional[str] = config.get("base_url")
        self.base_url_env: Optional[str] = config.get("base_url_env")
        self.api_key_env: str = config.get("api_key_env", "OPENAI_API_KEY")
        self.max_tokens: int = config.get("max_tokens", 4096)
        # azure_openai-only fields. Azure deployments need an api_version and
        # use `azure_endpoint` rather than the standard `base_url`. Per-worker
        # overrides via env: $<api_version_env>, $<base_url_env>.
        self.api_version: Optional[str] = config.get("api_version")
        self.api_version_env: Optional[str] = config.get("api_version_env")
        # google_vertex-only fields (None for other providers)
        self.vertex_project: Optional[str] = config.get("vertex_project")
        self.vertex_location: str = config.get("vertex_location", "global")
        self.credentials_path: Optional[str] = config.get("credentials_path")
        self.credentials_path_env: Optional[str] = config.get("credentials_path_env")
        self._client: Any = None  # lazily initialized

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        cache_prefix: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Send a prompt to the configured model and return the response.

        Args:
            prompt:      User-turn message text.
            system:      Optional system prompt.
            temperature: Sampling temperature (0.0–1.0+).
            max_tokens:  Maximum response tokens. If None, uses the value
                         from the model's configs/models.yaml entry (falls
                         back to 4096). IMPORTANT: the previous hard-coded
                         default of 4096 truncated long solutions for
                         hexagon_packing / cloudcast and similar tasks,
                         producing "invalid syntax" errors when the
                         truncated-mid-fence response was extracted as code.

        Returns:
            dict with keys:
                "text"              — assistant response string
                "prompt_tokens"     — input token count
                "completion_tokens" — output token count
        """
        if max_tokens is None:
            max_tokens = self.max_tokens
        if self.provider == "anthropic":
            return self._complete_anthropic(prompt, system, temperature, max_tokens, cache_prefix=cache_prefix)
        elif self.provider in ("openai", "openai_compatible", "azure_openai"):
            # OpenAI does automatic prefix caching server-side; cache_prefix is
            # informational here. We just concatenate it with the prompt.
            full = (cache_prefix + "\n\n" + prompt) if cache_prefix else prompt
            return self._complete_openai(full, system, temperature, max_tokens)
        elif self.provider == "google_vertex":
            # Vertex does implicit prefix caching for repeated leading content;
            # concatenate cache_prefix into the user turn (same shape as OpenAI).
            full = (cache_prefix + "\n\n" + prompt) if cache_prefix else prompt
            return self._complete_vertex(full, system, temperature, max_tokens)
        elif self.provider == "vertex_openai_maas":
            # Vertex AI Model Garden partner models (Qwen / Llama / Mistral /
            # Moonshot) are exposed via the OpenAI-compatible
            # /openapi/chat/completions endpoint, NOT the Gemini SDK.
            full = (cache_prefix + "\n\n" + prompt) if cache_prefix else prompt
            return self._complete_vertex_openai_maas(full, system, temperature, max_tokens)
        else:
            raise ValueError(f"Unknown provider: {self.provider!r}")

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------

    def _complete_anthropic(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        max_retries: int = 5,
        cache_prefix: Optional[str] = None,
    ) -> dict[str, Any]:
        import anthropic

        client = self._get_anthropic_client()
        # If a cache_prefix is provided AND it's large enough to be worth
        # caching (Anthropic requires >=1024 tokens for Opus to actually
        # cache; we approximate by char count to avoid an extra tokenizer
        # call — 1024 tokens ~= 4096 chars), build the user message as a
        # list of content blocks with cache_control on the prefix block.
        if cache_prefix and len(cache_prefix) >= 4096:
            user_content = [
                {
                    "type": "text",
                    "text": cache_prefix,
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": prompt},
            ]
        elif cache_prefix:
            # Too small to bother caching — concatenate inline.
            user_content = cache_prefix + "\n\n" + prompt
        else:
            user_content = prompt

        kwargs: dict[str, Any] = dict(
            model=self.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": user_content}],
        )
        if system:
            kwargs["system"] = system

        for attempt in range(max_retries):
            try:
                resp = client.messages.create(**kwargs)
                usage = resp.usage
                cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
                cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
                return {
                    "text": resp.content[0].text,
                    "prompt_tokens": usage.input_tokens + cache_creation + cache_read,
                    "completion_tokens": usage.output_tokens,
                    "cache_creation_input_tokens": cache_creation,
                    "cache_read_input_tokens": cache_read,
                }
            except anthropic.RateLimitError:
                wait = (2 ** attempt) * 10  # 10s, 20s, 40s, 80s, 160s
                print(f"[LLMClient] Anthropic rate limit (attempt {attempt+1}), waiting {wait}s")
                time.sleep(wait)
            except anthropic.APIStatusError as e:
                if e.status_code in (429, 529):
                    wait = (2 ** attempt) * 15
                    print(f"[LLMClient] Anthropic {e.status_code} (attempt {attempt+1}), waiting {wait}s")
                    time.sleep(wait)
                elif attempt < max_retries - 1:
                    print(f"[LLMClient] Anthropic API error {e.status_code}: {e.message} — retrying")
                    time.sleep(5)
                else:
                    raise
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[LLMClient] Unexpected error: {e} — retrying in 5s")
                    time.sleep(5)
                else:
                    raise

        raise RuntimeError(f"Anthropic API call failed after {max_retries} attempts")

    @staticmethod
    def _is_reasoning_model(model_id: str) -> bool:
        """
        Reasoning-tier OpenAI models (gpt-5*, o1*, o3*) reject `max_tokens`
        and ignore `temperature`. They expect `max_completion_tokens` instead
        and surface a separate `reasoning_tokens` count under usage.

        Excludes `*-chat-latest` aliases, which use the classic chat API
        contract (these accept temperature and max_tokens normally).
        """
        m = model_id.lower()
        if m.endswith("-chat-latest"):
            return False
        return m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3")

    def _complete_openai(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        max_retries: int = 10,
    ) -> dict[str, Any]:
        import openai

        client = self._get_openai_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {"model": self.model_id, "messages": messages}
        if self._is_reasoning_model(self.model_id):
            kwargs["max_completion_tokens"] = max_tokens
            # temperature is fixed at 1 for reasoning models — passing
            # anything else returns 400 unsupported_parameter.
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature

        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(**kwargs)
                usage = resp.usage
                completion = usage.completion_tokens
                # reasoning_tokens are billed as output (same line item as
                # visible output) but are already included in completion_tokens
                # by the API, so we return completion_tokens as-is and surface
                # reasoning_tokens separately for diagnostics.
                reasoning = 0
                details = getattr(usage, "completion_tokens_details", None)
                if details is not None:
                    reasoning = getattr(details, "reasoning_tokens", 0) or 0
                # gpt-oss-120b (and some Azure-hosted reasoning models) occasionally
                # return message.content == None when the model produced only
                # reasoning_content with no final answer, or hit a soft stop.
                # Don't pass None back to the agent — re.search on None crashes.
                # Retry the call (stochastic at temperature>0) before falling back.
                msg = resp.choices[0].message
                text = msg.content
                if not text and attempt < max_retries - 1:
                    wait = (2 ** attempt) * 3
                    finish_reason = getattr(resp.choices[0], "finish_reason", "?")
                    print(f"[LLMClient] Empty content from {self.model_id} "
                          f"(finish_reason={finish_reason}, attempt {attempt+1}), "
                          f"retrying in {wait}s")
                    time.sleep(wait)
                    continue
                if not text:
                    # Last attempt — fall back to reasoning_content if any so
                    # the agent at least sees the model's chain of thought
                    # (better than crashing or recording a fully empty step).
                    text = getattr(msg, "reasoning_content", "") or ""
                return {
                    "text": text,
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": completion,
                    "reasoning_tokens": reasoning,
                }
            except openai.RateLimitError:
                # Cap exponential backoff at 180s — at max_retries=10 the raw
                # exponential would hit 5120s on attempt 9, which blocks the
                # worker for ~3 hours instead of letting quota recover.
                wait = min((2 ** attempt) * 10, 180)
                print(f"[LLMClient] OpenAI rate limit (attempt {attempt+1}), waiting {wait}s")
                time.sleep(wait)
            except openai.APIStatusError as e:
                if e.status_code in (429, 529):
                    wait = min((2 ** attempt) * 15, 180)
                    print(f"[LLMClient] OpenAI {e.status_code} (attempt {attempt+1}), waiting {wait}s")
                    time.sleep(wait)
                elif attempt < max_retries - 1:
                    print(f"[LLMClient] OpenAI API error {e.status_code} — retrying")
                    time.sleep(5)
                else:
                    raise
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[LLMClient] Unexpected error: {e} — retrying in 5s")
                    time.sleep(5)
                else:
                    raise

        raise RuntimeError(f"OpenAI API call failed after {max_retries} attempts")

    def _complete_vertex(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        max_retries: int = 5,
    ) -> dict[str, Any]:
        client = self._get_vertex_client()

        config: dict[str, Any] = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            config["system_instruction"] = system

        for attempt in range(max_retries):
            try:
                resp = client.models.generate_content(
                    model=self.model_id,
                    contents=prompt,
                    config=config,
                )
                # Gemini 3 is a thinking model: when max_output_tokens is too
                # small, internal "thoughts" consume the whole budget and the
                # response has no visible parts (text == None / empty).
                # Surface this as a clear error rather than a silent empty reply.
                text = resp.text or ""
                usage = resp.usage_metadata
                prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
                visible_out = getattr(usage, "candidates_token_count", 0) or 0
                thoughts = getattr(usage, "thoughts_token_count", 0) or 0
                if not text:
                    # Read the actual finish_reason from the candidate — the
                    # token counters can be missing/zero for several distinct
                    # failure modes (MAX_TOKENS, SAFETY, recitation, etc.),
                    # so blaming "thoughts" without checking is misleading.
                    finish_reason = "?"
                    try:
                        finish_reason = str(getattr(resp.candidates[0], "finish_reason", "?"))
                    except Exception:
                        pass
                    if "MAX_TOKENS" in finish_reason:
                        raise RuntimeError(
                            f"Empty response from {self.model_id}: hit max_output_tokens "
                            f"({max_tokens}) — thoughts={thoughts}, visible={visible_out}, "
                            f"prompt={prompt_tokens}. Raise max_tokens (Vertex cap is 65537)."
                        )
                    if "SAFETY" in finish_reason or "BLOCK" in finish_reason or "RECITATION" in finish_reason:
                        raise RuntimeError(
                            f"Empty response from {self.model_id}: blocked "
                            f"(finish_reason={finish_reason}, prompt={prompt_tokens})."
                        )
                    raise RuntimeError(
                        f"Empty response from {self.model_id}: "
                        f"finish_reason={finish_reason}, thoughts={thoughts}, "
                        f"visible={visible_out}, max_tokens={max_tokens}."
                    )
                return {
                    "text": text,
                    "prompt_tokens": prompt_tokens,
                    # Thoughts are billed as output, so include them in the
                    # completion tally for cost reporting.
                    "completion_tokens": visible_out + thoughts,
                }
            except Exception as e:
                # google.genai bundles errors under a few classes; we don't
                # introspect the type and instead branch on the error message:
                #   - "hit max_output_tokens" — deterministic config issue,
                #     fail fast (retry with same config will hit the same wall).
                #   - other "Empty response" cases (MALFORMED_FUNCTION_CALL,
                #     RECITATION, blocked, etc.) — stochastic at temperature=1,
                #     a re-roll can succeed, so retry like a transient error.
                #   - everything else (503, network, etc.) — retry.
                msg = str(e)
                is_max_tokens = "hit max_output_tokens" in msg
                if is_max_tokens or attempt >= max_retries - 1:
                    raise
                wait = (2 ** attempt) * 5
                print(f"[LLMClient] Vertex error (attempt {attempt+1}): {e} — retrying in {wait}s")
                time.sleep(wait)

        raise RuntimeError(f"Vertex API call failed after {max_retries} attempts")

    def _complete_vertex_openai_maas(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        max_retries: int = 15,
    ) -> dict[str, Any]:
        """Vertex AI Model Garden partner MaaS via OpenAI-compatible endpoint.

        Used for Qwen / Llama / Mistral / Moonshot etc. The endpoint expects an
        OAuth2 Bearer token from the service account; AuthorizedSession refreshes
        it automatically every ~hour.
        """
        session, url = self._get_vertex_maas_session()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        for attempt in range(max_retries):
            try:
                r = session.post(url, json=payload, timeout=180)
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = (2 ** attempt) * 5
                    print(f"[LLMClient] Vertex MaaS network error (attempt {attempt+1}): {e} — retrying in {wait}s")
                    time.sleep(wait)
                    continue
                raise

            if r.status_code == 200:
                data = r.json()
                choice = data["choices"][0]
                msg = choice.get("message", {})
                text = msg.get("content")
                # Reasoning-tier MaaS models (e.g. Kimi K2 Thinking) put their
                # CoT in reasoning_content and may leave content empty when the
                # token budget is consumed by thoughts. Mirror the openai handler
                # behavior: retry once on empty, then fall back to CoT.
                if not text and attempt < max_retries - 1:
                    wait = (2 ** attempt) * 3
                    finish_reason = choice.get("finish_reason", "?")
                    print(f"[LLMClient] Empty content from {self.model_id} "
                          f"(finish_reason={finish_reason}, attempt {attempt+1}), "
                          f"retrying in {wait}s")
                    time.sleep(wait)
                    continue
                if not text:
                    text = msg.get("reasoning_content") or ""
                usage = data.get("usage") or {}
                return {
                    "text": text,
                    "prompt_tokens": usage.get("prompt_tokens", 0) or 0,
                    "completion_tokens": usage.get("completion_tokens", 0) or 0,
                }

            if r.status_code in (429, 503) and attempt < max_retries - 1:
                # MaaS partner models share Dynamic Shared Quota — 429 bursts
                # are usually short (~30-60s). Start with 10s, double, cap at
                # 300s. 15 attempts ≈ 50min total backoff but first 5 attempts
                # only burn ~5min — quick recovery on transient bursts.
                wait = min((2 ** attempt) * 10, 300)
                print(f"[LLMClient] Vertex MaaS {r.status_code} (attempt {attempt+1}/{max_retries}), "
                      f"waiting {wait}s — body: {r.text[:200]}")
                time.sleep(wait)
                continue

            # Non-retryable error (404 wrong model id, 400 malformed, etc.)
            raise RuntimeError(
                f"Vertex MaaS {self.model_id} returned {r.status_code}: {r.text[:500]}"
            )

        raise RuntimeError(f"Vertex MaaS call failed after {max_retries} attempts")

    # ------------------------------------------------------------------
    # Lazy client construction
    # ------------------------------------------------------------------

    def _get_anthropic_client(self):
        if self._client is None:
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def _load_vertex_service_account(self) -> tuple[Any, str]:
        """Load the service-account key, return (Credentials, project_id).

        Resolution order for the key path:
          $credentials_path_env > config.credentials_path > $GOOGLE_APPLICATION_CREDENTIALS.
        """
        from google.oauth2 import service_account

        key_path: Optional[str] = None
        if self.credentials_path_env:
            key_path = os.environ.get(self.credentials_path_env)
        if not key_path:
            key_path = self.credentials_path
        if not key_path:
            key_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not key_path:
            raise RuntimeError(
                f"vertex model {self.model_name!r}: no credentials path. "
                "Set credentials_path in models.yaml, the credentials_path_env "
                "var, or $GOOGLE_APPLICATION_CREDENTIALS."
            )

        path = Path(key_path)
        if not path.exists():
            raise FileNotFoundError(f"Service-account key not found: {path}")

        with path.open() as fh:
            info = json.load(fh)
        # Some service-account keys omit "type" and may double-escape \n in
        # private_key — google.oauth2 needs both fixed.
        info.setdefault("type", "service_account")
        if "\\n" in info.get("private_key", ""):
            info["private_key"] = info["private_key"].replace("\\n", "\n")

        project = self.vertex_project or info.get("project_id")
        if not project:
            raise RuntimeError(
                f"vertex model {self.model_name!r}: no project. "
                "Set vertex_project in models.yaml or include project_id in the key."
            )

        creds = service_account.Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        return creds, project

    def _get_vertex_client(self):
        if self._client is None:
            from google import genai

            creds, project = self._load_vertex_service_account()
            self._client = genai.Client(
                vertexai=True,
                project=project,
                location=self.vertex_location,
                credentials=creds,
            )
        return self._client

    def _get_vertex_maas_session(self):
        """Authorized session that auto-refreshes the SA OAuth token (~1h TTL)."""
        if self._client is None:
            from google.auth.transport.requests import AuthorizedSession

            creds, project = self._load_vertex_service_account()
            session = AuthorizedSession(creds)
            host = (
                "aiplatform.googleapis.com"
                if self.vertex_location == "global"
                else f"{self.vertex_location}-aiplatform.googleapis.com"
            )
            url = (
                f"https://{host}/v1/projects/{project}"
                f"/locations/{self.vertex_location}/endpoints/openapi/chat/completions"
            )
            self._client = (session, url)
        return self._client

    def _get_openai_client(self):
        if self._client is None:
            import openai
            # Per-worker base_url override via env (used by the pilot pool).
            env_url = None
            if self.base_url_env:
                env_url = os.environ.get(self.base_url_env)
            base_url = env_url or self.base_url

            if self.provider == "azure_openai":
                # Azure deployments need (api_version, azure_endpoint, api_key).
                # The api_version may also be overridden per-slot via env.
                env_ver = None
                if self.api_version_env:
                    env_ver = os.environ.get(self.api_version_env)
                api_version = env_ver or self.api_version
                if not api_version:
                    raise RuntimeError(
                        f"azure_openai model {self.model_name!r}: api_version required "
                        "(set api_version in models.yaml or $<api_version_env>)."
                    )
                if not base_url:
                    raise RuntimeError(
                        f"azure_openai model {self.model_name!r}: base_url (azure_endpoint) "
                        "required (set base_url in models.yaml or $<base_url_env>)."
                    )
                self._client = openai.AzureOpenAI(
                    api_version=api_version,
                    azure_endpoint=base_url,
                    api_key=os.environ.get(self.api_key_env),
                )
            else:
                kwargs: dict[str, Any] = {
                    "api_key": os.environ.get(self.api_key_env),
                }
                if base_url:
                    kwargs["base_url"] = base_url
                self._client = openai.OpenAI(**kwargs)
        return self._client

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_model_config(model_name: str, config_path: str) -> dict[str, Any]:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Models config not found: {config_path}")
        with path.open() as fh:
            data = yaml.safe_load(fh)
        for entry in data.get("models", []):
            if entry["name"] == model_name:
                return entry
        raise KeyError(
            f"Model {model_name!r} not found in {config_path}. "
            f"Available: {[m['name'] for m in data.get('models', [])]}"
        )
