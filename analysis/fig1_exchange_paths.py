"""fig1_exchange_paths.py - exchange paths and zigzag order in one FePS3 layer (geometry: Jena et al., PRB 113, 064429 (2026), Table I)."""
import os, sys, itertools
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib as mpl, matplotlib.pyplot as plt
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'analysis'))
from figstyle import apply_figure_style
apply_figure_style(sizes=(8, 7, 6))
a, b = 5.931, 10.309
base = [((0.5, 0.16745), 1), ((0.0, 0.66745), -1), ((0.5, 0.83255), -1), ((0.0, 0.33255), 1)]
sites = [(np.array([(fx + i) * a, (fy + j) * b]), s) for i, j in itertools.product(range(-1, 4), range(-1, 3)) for (fx, fy), s in base]
P = np.array([p for p, _ in sites]); Sg = np.array([s for _, s in sites])
xmin, xmax, ymin, ymax = -0.3 * a, 3.3 * a, -0.05 * b, 2.05 * b
k = (P[:, 0] > xmin) & (P[:, 0] < xmax) & (P[:, 1] > ymin) & (P[:, 1] < ymax); P, Sg = P[k], Sg[k]
D = np.linalg.norm(P[:, None] - P[None], axis=2)
ref = int(np.argmin(np.linalg.norm(P - np.array([1.5 * a, 1.16745 * b]), axis=1))); r0 = P[ref]
C = {'J1a': '#E69F00', 'J1b': '#0072B2', 'J2': '#009E73', 'J3': '#CC79A7'}
fig, ax = plt.subplots(figsize=(3.4, 2.9))
for i in range(len(P)):
    for j in range(i + 1, len(P)):
        if D[i, j] < 3.5:
            fm = D[i, j] < 3.43 and Sg[i] == Sg[j]
            ax.plot(*zip(P[i], P[j]), color='0.55' if fm else '0.82', lw=1.6 if fm else 0.8, zorder=1)
ends = {}
for j in range(len(P)):
    d = D[ref, j]
    if j == ref: continue
    if d < 3.43: key, ls, lw = 'J1a', '-', 2.6
    elif d < 3.5: key, ls, lw = 'J1b', '-', 2.6
    elif 5.8 < d < 6.0: key, ls, lw = 'J2', (0, (3, 1.5)), 1.4
    elif 6.8 < d < 6.95: key, ls, lw = 'J3', (0, (1, 1.2)), 1.6
    else: continue
    ax.plot(*zip(r0, P[j]), color=C[key], ls=ls, lw=lw, zorder=2, solid_capstyle='round'); ends.setdefault(key, []).append(P[j])
assert {k: len(v) for k, v in ends.items()} == {'J1a': 2, 'J1b': 1, 'J2': 6, 'J3': 3}
up = Sg > 0
ax.scatter(P[up, 0], P[up, 1], s=34, c='k', edgecolors='k', lw=0.8, zorder=4, label=r'$S\parallel+c^{*}$')
ax.scatter(P[~up, 0], P[~up, 1], s=34, c='white', edgecolors='k', lw=0.9, zorder=4, label=r'$S\parallel-c^{*}$')
ax.scatter(*r0, s=110, facecolors='none', edgecolors='k', lw=1.0, zorder=5)
pick = lambda key, f: sorted(ends[key], key=f)[0]
pos = {'J1a': pick('J1a', lambda p: -p[0]) + np.array([0.3, 1.25]), 'J1b': (r0 + pick('J1b', lambda p: p[1])) / 2 + np.array([-1.3, 0.0]),
       'J2': pick('J2', lambda p: (-p[0], abs(p[1] - r0[1]))) + np.array([1.0, 0.9]), 'J3': pick('J3', lambda p: (-p[0], p[1])) + np.array([1.25, -0.15])}
txt = {'J1a': r'$J_1^{a}$', 'J1b': r'$J_1^{b}$', 'J2': r'$J_2$', 'J3': r'$J_3$'}
for key, m in pos.items():
    ax.text(*m, txt[key], color=C[key], fontsize=8, ha='center', va='center', zorder=6, bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none'))
ax.add_patch(mpl.patches.Rectangle((a, 0), a, b, fill=False, ec='0.35', lw=0.7, ls=(0, (4, 2)), zorder=0))
x0, y0 = -0.15 * a, 0.0
ax.annotate('', xy=(x0 + 3.0, y0), xytext=(x0, y0), arrowprops=dict(arrowstyle='->', lw=0.9, color='k'))
ax.annotate('', xy=(x0, y0 + 3.0), xytext=(x0, y0), arrowprops=dict(arrowstyle='->', lw=0.9, color='k'))
ax.text(x0 + 3.3, y0, r'$\mathbf{a}$', va='center', fontsize=8); ax.text(x0, y0 + 3.5, r'$\mathbf{b}$', ha='center', fontsize=8)
ax.set_aspect('equal'); ax.set_xlim(xmin - 0.5, xmax + 1.2); ax.set_ylim(ymin - 0.8, ymax + 0.3); ax.axis('off')
ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.10), ncol=2, frameon=False, handletextpad=0.2, columnspacing=1.2, fontsize=7)
fig.savefig('Sekil_J_orgu.png', dpi=300, bbox_inches='tight'); fig.savefig('Sekil_J_orgu.pdf', bbox_inches='tight')
