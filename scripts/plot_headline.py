"""Headline figure: the two regimes side by side.

Left, 27B base (success at ceiling): turns saved vs arm A, results/summary.json.
Right, matched 7B base (success has headroom): success advantage over A,
results/summary_h3.json. Both on test-indist, clustered-bootstrap 95% CIs.

    python scripts/plot_headline.py   # -> results/figures/fig6_two_regimes.png
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COLORS = {'B': '#5b8def', 'C': '#e8883a', 'D': '#2fae94', 'E': '#d9aa2b', 'F': '#c46fb0'}
NAMES = {'B': 'B raw examples', 'C': 'C episodic retrieval', 'D': 'D playbook (ACE)',
         'E': 'E QLoRA weights', 'F': 'F weights + playbook'}
INK, MUTED, GRID = '#1f2328', '#6b7280', '#e5e7eb'


def panel(ax, rows, title, xlabel):
    for i, (arm, mean, lo, hi, n) in enumerate(rows):
        ax.plot([lo, hi], [i, i], color=COLORS[arm], linewidth=2, solid_capstyle='round')
        ax.plot(mean, i, 'o', color=COLORS[arm], markersize=8, markeredgecolor='white', markeredgewidth=2)
        ax.annotate(f'{mean:+.2f}  (n={n})', (hi, i), xytext=(6, -3), textcoords='offset points',
                    color=MUTED, fontsize=8)
    ax.axvline(0, color=MUTED, linewidth=1, linestyle='--')
    ax.set_yticks(range(len(rows)), [NAMES[r[0]] for r in rows])
    ax.invert_yaxis()
    ax.set_title(title, color=INK, fontsize=10, loc='left')
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    pilot = json.loads((ROOT/'results/summary.json').read_text())['H1b_turns_saved_vs_A']
    h3 = json.loads((ROOT/'results/summary_h3.json').read_text())['H1_advantage_over_A']
    left = [(arm, c['turns_saved_vs_A'], c['ci_lo'], c['ci_hi'], c['n_pairs'])
            for arm in 'BCD' for c in [pilot[f'{arm}|test-indist']]]
    right = [(arm, c['advantage'], c['ci_lo'], c['ci_hi'], c['n_pairs']) for arm, c in h3.items()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw=dict(width_ratios=[1, 1.25]))
    panel(axes[0], left, '27B base: every arm solves everything', 'turns saved vs no memory (A)')
    panel(axes[1], right, 'Matched 7B base: success has headroom', 'success rate minus no memory (A)')
    axes[0].set_xlim(-0.1, 2.3)
    axes[1].set_xlim(-0.55, 1.05)
    fig.text(0.01, 0.01, 'In-distribution test split. Dots: point estimate. Lines: 95% bootstrap CI, '
             'resampled over the 5 challenge families.', color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = ROOT/'results/figures/fig6_two_regimes.png'
    fig.savefig(out, dpi=180)
    print(out)


if __name__ == '__main__':
    main()
