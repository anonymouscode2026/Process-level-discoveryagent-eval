import json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[2]

with open(_PKG_ROOT / 'analysis/harness_by_task/plot_payload.json') as f:
    p = json.load(f)

LEVELS = p['levels']
FAMILIES_LIST = p['families']
fam_data = p['fam_data']
quad_data = p['quad_data']
quad_keys = p['quad_keys']
OUT_DIR = p['out_dir']

# ── Heatmap 1: Family × Level ─────────────────────────────────────────────────
data = np.array([[fam_data[fam][lv]['mean'] for lv in LEVELS] for fam in FAMILIES_LIST])
stds = np.array([[fam_data[fam][lv]['std']  for lv in LEVELS] for fam in FAMILIES_LIST])

fig, ax = plt.subplots(figsize=(7.5, 4))
im = ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=0.38, vmax=0.60)
ax.set_xticks(range(len(LEVELS)))
ax.set_xticklabels(LEVELS, fontsize=13, fontweight='bold')
ax.set_yticks(range(len(FAMILIES_LIST)))
ax.set_yticklabels(FAMILIES_LIST, fontsize=11)
ax.set_title('Mean Rank-Normalized Score: Task Family × Agency Level', fontsize=12, pad=10)
ax.set_xlabel('Agency Level', fontsize=11)
ax.set_ylabel('Task Family', fontsize=11)

for i, fam in enumerate(FAMILIES_LIST):
    for j, lv in enumerate(LEVELS):
        val = data[i, j]
        std = stds[i, j]
        color = 'white' if val < 0.43 or val > 0.58 else 'black'
        ax.text(j, i, f'{val:.3f}\n±{std:.3f}', ha='center', va='center',
                fontsize=9, color=color)

plt.colorbar(im, ax=ax, label='Mean rank-normalized score', shrink=0.85)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/family_x_level_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved family_x_level_heatmap.png")

# ── Heatmap 2: Quadrant × Level ──────────────────────────────────────────────
quad_labels = [k.replace('|', '/') for k in quad_keys]
data2 = np.array([[quad_data[k][lv]['mean'] for lv in LEVELS] for k in quad_keys])
stds2 = np.array([[quad_data[k][lv]['std']  for lv in LEVELS] for k in quad_keys])

fig, ax = plt.subplots(figsize=(7.5, 6))
im = ax.imshow(data2, aspect='auto', cmap='RdYlGn', vmin=0.38, vmax=0.65)
ax.set_xticks(range(len(LEVELS)))
ax.set_xticklabels(LEVELS, fontsize=13, fontweight='bold')
ax.set_yticks(range(len(quad_labels)))
ax.set_yticklabels([f'alg={q.split("/")[0]}, repr={q.split("/")[1]}' for q in quad_labels], fontsize=10)
ax.set_title('Mean Rank-Normalized Score: Capability Quadrant × Agency Level', fontsize=12, pad=10)
ax.set_xlabel('Agency Level', fontsize=11)
ax.set_ylabel('Capability Demand (alg, repr)', fontsize=11)

for i, k in enumerate(quad_keys):
    for j, lv in enumerate(LEVELS):
        val = data2[i, j]
        std = stds2[i, j]
        color = 'white' if val < 0.43 or val > 0.60 else 'black'
        ax.text(j, i, f'{val:.3f}\n±{std:.3f}', ha='center', va='center',
                fontsize=8, color=color)

plt.colorbar(im, ax=ax, label='Mean rank-normalized score', shrink=0.85)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/quadrant_x_level_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved quadrant_x_level_heatmap.png")
