"""sekil_uc_kisit.py - Makale 1 Sekil: uc kisit. Girdi: Tablo_uc_kisit.csv, MF_kisitlar.csv (F_BC). Cikti: Sekil_uc_kisit.png/.pdf."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

def make_figure(TAB, MF, apply_figure_style, panel_letter, out='Sekil_uc_kisit'):
    apply_figure_style()
    rank7 = ['Wildes20_r4_x2', 'Wildes20_r2_x2', 'Lancon_x2', 'Yang_U36', 'Amir_U222', 'Olsen_U2', 'Chat_U25']
    col = {'Wildes20_r4_x2': '#67000d', 'Wildes20_r2_x2': '#ef3b2c', 'Lancon_x2': '#c0392b', 'Yang_U36': '#7f7f7f',
           'Amir_U222': '#595959', 'Olsen_U2': '#a6a6a6', 'Chat_U25': '#8c8c8c'}
    mk = {'Wildes20_r4_x2': 'D', 'Wildes20_r2_x2': 'D', 'Lancon_x2': 'D', 'Yang_U36': 's', 'Amir_U222': 'v', 'Olsen_U2': '^', 'Chat_U25': 'P'}
    short = {'Wildes20_r4_x2': 'Wildes 4', 'Wildes20_r2_x2': 'Wildes 2', 'Lancon_x2': 'Lançon', 'Yang_U36': 'Yang',
             'Amir_U222': 'Amirabbasi', 'Olsen_U2': 'Olsen', 'Chat_U25': 'Chaturvedi'}
    inv = {'Wildes20_r4_x2': 'W4pub', 'Wildes20_r2_x2': 'W2pub', 'Olsen_U2': 'OlsenD', 'Yang_U36': 'YangD', 'Amir_U222': 'AmirDB'}
    EXP = {'Jena_yeni': ('Jena 2026', '0.70'), 'Joy1992': ('Joy 1992', '0.85')}
    T = TAB.set_index('set'); x = np.arange(len(rank7)); dEexp = float(T.loc['Jena_yeni', 'dE_meV_Fe'])
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.0)); (a, b), (c, d) = axs

    def mark(ax, xx, y, s, kind, yerr=None):
        fc = {'I': 'white', 'H': col[s], 'P': col[s]}[kind]; ec = {'I': col[s], 'H': col[s], 'P': 'k'}[kind]
        ax.errorbar(xx, y, yerr=yerr, fmt=mk[s], ms=5 if kind != 'P' else 5.5, mfc=fc, mec=ec, mew=1.0 if kind != 'P' else 1.3,
                    ecolor=col[s], elinewidth=0.8, capsize=0, zorder=3)

    def band(ax, lo, hi, c, x0=None, x1=None):
        if x0 is None: ax.axhspan(lo, hi, color=c, alpha=0.55, lw=0, zorder=0)
        else: ax.add_patch(plt.Rectangle((x0, lo), x1 - x0, hi - lo, color=c, alpha=0.55, lw=0, zorder=1))

    for e, (lbl, cc) in EXP.items(): band(a, T.loc[e, 'TN_bant_alt_K'], T.loc[e, 'TN_bant_ust_K'], cc)
    for i, s in enumerate(rank7):
        a.plot([i, i], [T.loc[s, 'TN_Heis_K'], T.loc[s, 'TN_Ising_K']], color=col[s], lw=1.2, zorder=2)
        mark(a, i, T.loc[s, 'TN_Ising_K'], s, 'I', T.loc[s, 'sTN_Ising_K']); mark(a, i, T.loc[s, 'TN_Heis_K'], s, 'H', T.loc[s, 'sTN_Heis_K'])
        if s in inv: mark(a, i + 0.28, T.loc[inv[s], 'TN_Heis_K'], s, 'P', T.loc[inv[s], 'sTN_Heis_K'])
    a.set_ylabel(r'$T_\mathrm{N}$ (K)')
    for e, (lbl, cc) in EXP.items(): band(b, T.loc[e, 'F_exp_ana_min'], T.loc[e, 'F_exp_ana_max'], cc)
    for i, s in enumerate(rank7):
        mark(b, i - 0.1, T.loc[s, 'F0_ana_I'], s, 'I', T.loc[s, 'F0_ana_I_se']); mark(b, i + 0.1, T.loc[s, 'F0_ana_H'], s, 'H', T.loc[s, 'F0_ana_H_se'])
        b.plot(i, MF.set_index('set').loc[s, 'F_BC'], '_', color=col[s], ms=8, mew=1.2, zorder=3)
        if s in inv: mark(b, i + 0.3, T.loc[inv[s], 'F0_ana_H'], s, 'P', T.loc[inv[s], 'F0_ana_H_se'])
    b.set_ylabel(r'$F=[J(\mathbf{Q})-J(0)]\,\Delta\chi_\parallel$'); b.set_ylim(0, 1.25)
    wl = [('dar', '0.90–1.05'), ('ana', '0.83–1.10'), ('genis', '0.75–1.20')]
    for k, (w, lbl) in enumerate(wl):
        for j, (e, (el, cc)) in enumerate(EXP.items()):
            band(c, T.loc[e, f'F_exp_{w}_min'], T.loc[e, f'F_exp_{w}_max'], cc, k - 0.34 + 0.34 * j, k + 0.34 * j)
        for i, s in enumerate(rank7):
            xx = k - 0.27 + 0.09 * i
            for m in ('I', 'H'):
                v = T.loc[s, f'F0_{w}_{m}']
                if np.isfinite(v): mark(c, xx + (0.0 if m == 'I' else 0.03), v, s, m)
    c.set_xticks(range(3)); c.set_xticklabels([l for _, l in wl]); c.set_xlabel(r'window $T/T_\mathrm{N}$'); c.set_ylabel(r'$F$')
    c.set_ylim(0, 1.25); c.set_xlim(-0.5, 2.5)
    d.axhline(dEexp, color='k', lw=1.0, zorder=1)
    for i, s in enumerate(rank7): mark(d, i, T.loc[s, 'dE_meV_Fe'], s, 'H')
    d.set_yscale('log'); d.set_ylabel(r'$E_\mathrm{FM}-E_\mathrm{zz}$ (meV/Fe)'); d.set_ylim(5, 70)
    d.text(6.45, dEexp * 1.08, 'exp. (Wildes 2020: 39.35 + 41.3 T)', fontsize=6, ha='right', va='bottom')
    d.yaxis.set_major_locator(FixedLocator([5, 10, 20, 50])); d.yaxis.set_major_formatter(FixedFormatter(['5', '10', '20', '50'])); d.yaxis.set_minor_locator(NullLocator())
    d2 = d.twinx(); d2.set_yscale('log'); lo, hi = d.get_ylim(); d2.set_ylim(lo / (4 * 0.0578838), hi / (4 * 0.0578838))
    d2.yaxis.set_major_locator(FixedLocator([30, 50, 100, 200])); d2.yaxis.set_major_formatter(FixedFormatter(['30', '50', '100', '200'])); d2.yaxis.set_minor_locator(NullLocator())
    d2.set_ylabel('direct spin-flip field (T)'); d2.spines['right'].set_visible(True)
    for ax in (a, b, d):
        ax.set_xticks(x); ax.set_xticklabels([short[s] for s in rank7], rotation=35, ha='right'); ax.set_xlim(-0.5, 6.6)
    h = [Line2D([], [], marker='o', ls='', mfc='white', mec='0.3', label='Ising (Blume–Capel)'),
         Line2D([], [], marker='o', ls='', mfc='0.3', mec='0.3', label='Heisenberg + D'),
         Line2D([], [], marker='o', ls='', mfc='0.3', mec='k', mew=1.3, label='published H'),
         Line2D([], [], marker='_', ls='', color='0.3', ms=8, mew=1.2, label='mean field (BC)')] + \
        [Patch(color=cc, alpha=0.55, lw=0, label=f'exp. {el}') for el, cc in EXP.values()]
    fig.legend(handles=h, loc='upper center', ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.01), columnspacing=1.0, handletextpad=0.4)
    for ax, l in zip((a, b, c, d), 'abcd'): panel_letter(ax, l)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out + '.png', dpi=300); fig.savefig(out + '.pdf')
    return fig
