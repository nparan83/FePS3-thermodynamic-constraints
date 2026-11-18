import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'analysis')); sys.path.insert(0, os.path.join(REPO, 'mc', 'lib'))
def P(f):
    for d in ('data/experiment', 'data/derived', 'mc/lib', 'analysis'):
        p = os.path.join(REPO, d, f)
        if os.path.exists(p): return p
    raise FileNotFoundError(f)

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from figstyle import apply_figure_style, panel_letter, META_GREY

sf = pd.read_csv(P('spinflip_alanlari.csv'))
LSWT_kontrol = pd.read_csv(P('LSWT_kontrol.csv'))

sf = sf.set_index('set')

rank7 = ['Wildes20_r4_x2', 'Wildes20_r2_x2', 'Lancon_x2', 'Yang_U36', 'Amir_U222', 'Olsen_U2', 'Chat_U25']
col = {'Wildes20_r4_x2': '#67000d', 'Wildes20_r2_x2': '#ef3b2c', 'Lancon_x2': '#c0392b', 'Yang_U36': '#7f7f7f',
       'Amir_U222': '#595959', 'Olsen_U2': '#a6a6a6', 'Chat_U25': '#8c8c8c'}
short = {'Wildes20_r4_x2': 'Wildes 4', 'Wildes20_r2_x2': 'Wildes 2', 'Lancon_x2': 'Lançon', 'Yang_U36': 'Yang',
         'Amir_U222': 'Amirabbasi', 'Olsen_U2': 'Olsen', 'Chat_U25': 'Chaturvedi'}
mk = {'Wildes20_r4_x2': 'D', 'Wildes20_r2_x2': 'D', 'Lancon_x2': 'D', 'Yang_U36': 's',
      'Amir_U222': 'v', 'Olsen_U2': '^', 'Chat_U25': 'P'}

plt.close('all')
apply_figure_style()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw=dict(width_ratios=[1.25, 1]))

ordr = rank7[::-1]
for i, s in enumerate(ordr):
    b = sf.loc[s, 'B_direct_T']
    a1.plot([20, b], [i, i], color=col[s], lw=1)
    a1.plot(b, i, mk[s], color=col[s], ms=5)

a1.axvspan(39.35, 41.3, color='0.75', lw=0, zorder=0)
a1.text(43, len(ordr) - 0.6, 'measured\n39.35, 41.3 T', fontsize=6, va='top')
a1.set_xscale('log')
a1.set_xlim(20, 260)
a1.set_xticks([20, 40, 100, 200])
a1.set_xticklabels(['20', '40', '100', '200'])
a1.minorticks_off()
a1.set_yticks(range(len(ordr)))
a1.set_yticklabels([short[s] for s in ordr])
a1.set_xlabel('zigzag → FM field, Ising limit, T = 0 (T)')

lw = LSWT_kontrol[LSWT_kontrol.anisotropy_term.str.contains('classical')].reset_index(drop=True)
for i, r in lw.iterrows():
    c = '#c0392b' if 'x2' in r.set else '0.6'
    a2.plot([i, i], [r.gap, r.top], color=c, lw=4, solid_capstyle='butt')

a2.axhspan(15, 17, color='0.8', lw=0, zorder=0)
a2.axhline(40, color='k', ls='--', lw=0.8)
a2.text(len(lw) - 0.5, 40.6, 'powder band top', fontsize=6, ha='right')

labs = ['Lançon ×1', 'Lançon ×2', 'Wildes 4 ×1', 'Wildes 4 ×2', 'Wildes 2 ×1', 'Wildes 2 ×2']
a2.set_xticks(range(len(lw)))
a2.set_xticklabels(labs, rotation=35, ha='right', fontsize=6)
a2.set_ylabel('LSWT magnon band (meV)')
a2.set_ylim(0, 48)
a2.set_xlim(-0.6, len(lw) - 0.4)

for t in a2.texts:
    if t.get_text() == 'INS gap':
        t.set_position((-0.5, 17.4))
        t.set_ha('left')

a2.text(-0.5, 17.4, 'INS gap', fontsize=6, ha='left')

panel_letter(a1, 'a')
panel_letter(a2, 'b')
fig.tight_layout()
fig.savefig('Sekil_spinflip_A.png', dpi=300)
fig.savefig('Sekil_spinflip_A.pdf')