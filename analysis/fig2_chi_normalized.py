import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'analysis')); sys.path.insert(0, os.path.join(REPO, 'mc', 'lib'))
def P(f):
    for d in ('data/experiment', 'data/derived', 'mc/lib', 'analysis'):
        p = os.path.join(REPO, d, f)
        if os.path.exists(p): return p
    raise FileNotFoundError(f)

import os
import json
import tarfile
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from figstyle import apply_figure_style, panel_letter, META_GREY

# Load data files
def rd(v):
    return pd.read_csv(v, sep=';', decimal=',', header=None, names=['T', 'y'], encoding='utf-8-sig')

joy = rd(P('Joy_1992_fig3_X_m_vs_K.csv'))
ja = rd(P('new_Fig3a_Data_0.05T_Dik.csv'))
jb = rd(P('new_Fig3b_Data__0.05T_Paralel.csv'))

mf = pd.read_csv(P('MF_kisitlar.csv'))

# Load MC v7 data
od0 = os.path.join(REPO, 'mc')
p = os.path.join(od0, 'mc_v7', 'mc_v7_sonuc.tar.gz')
os.makedirs('v7', exist_ok=True)
with tarfile.open(p) as t:
    t.extractall('v7')
R7 = pd.DataFrame([json.loads(l) for l in open('v7/mc7_raw.jsonl')])

# Load MC v6 data
p6 = os.path.join(od0, 'mc_v6', 'mc_v6_sonuc.tar.gz')
os.makedirs('v6', exist_ok=True)
with tarfile.open(p6) as t:
    names = [m.name for m in t.getmembers()]
    t.extractall('v6')
f6 = [n for n in names if n.endswith('.jsonl')]
R6 = pd.concat([pd.DataFrame([json.loads(l) for l in open('v6/' + n)]).assign(src=n) for n in f6])

# Load Ising data from v3, v4, v5
RI = []
for v in ('3', '4', '5'):
    with tarfile.open(os.path.join(od0, f'mc_v{v}', f'mc_v{v}_sonuc.tar.gz')) as t:
        for m in t.getmembers():
            if m.name.endswith('.jsonl'):
                d = pd.DataFrame([json.loads(l) for l in t.extractfile(m).read().decode().splitlines() if l.strip()])
                d['src'] = 'v' + v
                RI.append(d)
RI = pd.concat(RI, ignore_index=True)
if 'L' not in RI:
    RI['L'] = 24
RI['L'] = RI['L'].fillna(24)

rank7 = ['Wildes20_r4_x2', 'Wildes20_r2_x2', 'Lancon_x2', 'Yang_U36', 'Amir_U222', 'Olsen_U2', 'Chat_U25']

col = {'Wildes20_r4_x2': '#67000d', 'Wildes20_r2_x2': '#ef3b2c', 'Lancon_x2': '#c0392b', 'Yang_U36': '#7f7f7f',
       'Amir_U222': '#595959', 'Olsen_U2': '#a6a6a6', 'Chat_U25': '#8c8c8c'}
short = {'Wildes20_r4_x2': 'Wildes 4', 'Wildes20_r2_x2': 'Wildes 2', 'Lancon_x2': 'Lançon', 'Yang_U36': 'Yang',
         'Amir_U222': 'Amirabbasi', 'Olsen_U2': 'Olsen', 'Chat_U25': 'Chaturvedi'}

# Compute dJ from MF_kisitlar.csv
dJ = dict(zip(mf.set, mf.dJ))

# Compute chi_MF normalization
muB_cgs = 9.2740100783e-21
NA = 6.02214076e23
eV_erg = 1.602176634e-12
Mmol = 183.0
Msat_mol = 4 * muB_cgs * NA
dEexp = 9.3367
dE_mol = dEexp * 1e-3 * eV_erg * NA
chiMF_mol_direct = Msat_mol**2 / (2 * dE_mol)

# Compute experimental T_N via Fisher d(chi_par T)/dT max
WIN = {'ana': (0.830, 1.103), 'dar': (0.90, 1.05), 'genis': (0.75, 1.20)}

def tn_fisher(T, chi, lo=100, hi=135):
    g = pd.Series(chi, index=T).groupby(level=0).mean().sort_index()
    t = g.index.values
    y = g.values * t
    tt = np.linspace(lo, hi, 3501)
    yy = np.interp(tt, t, y)
    dy = np.gradient(yy, tt)
    return tt[np.argmax(dy)]

# Prepare experimental data
ja_mol = ja.assign(chi=ja.y / 500 * Mmol)
sets_exp = {
    'Jena_yeni': (ja['T'].values, ja.y.values / 500 * Mmol, chiMF_mol_direct),
    'Joy1992': (joy['T'].values, joy.y.values, chiMF_mol_direct)
}

FX_rows = []
for n, (T_, c_, nm) in sets_exp.items():
    tf = tn_fisher(T_, c_)
    FX_rows.append({'veri': n, 'TN_Fisher': tf})
FX = pd.DataFrame(FX_rows)

# Build TAB2 with TN values from Binder analysis
B7 = R7[R7.kind == 'B']

def pkfit(T, y, npt=2):
    T = np.asarray(T)
    y = np.asarray(y)
    j = int(np.nanargmax(y))
    lo, hi = max(j - npt, 0), min(j + npt + 1, len(T))
    if hi - lo < 3:
        return T[j]
    pc = np.polyfit(T[lo:hi], y[lo:hi], 2)
    t = -pc[1] / (2 * pc[0]) if pc[0] < 0 else T[j]
    return t if T[lo] <= t <= T[hi - 1] else T[j]

rows_pk = []
for (mdl, s), g in B7.groupby(['model', 'set']):
    r = dict(model=mdl, set=s)
    for L in (24, 36, 48):
        d = g[g.L == L].sort_values('T')
        r[f'C{L}'] = pkfit(d['T'], d['C'])
        r[f'X{L}'] = pkfit(d['T'], d['chizz'])
    rows_pk.append(r)
PK = pd.DataFrame(rows_pk)
PK['sTN'] = PK.apply(lambda r: 0.5 * (max(r.C36, r.C48, r.X36, r.X48) - min(r.C36, r.C48, r.X36, r.X48)), axis=1)
pkd = {(r.model, r.set): r for r in PK.itertuples()}

# Build TAB2 for TN values
TAB2_rows = []
for s in rank7:
    r_I = pkd[('I', s)]
    r_H = pkd[('H', s)]
    TAB2_rows.append(dict(set=s, TN_Ising_K=r_I.C48, TN_Heis_K=r_H.C48))
TAB2 = pd.DataFrame(TAB2_rows).set_index('set')

# Build CH (chi normalized data)
CH = []

for mdl in ('Ising', 'Heis'):
    for s in rank7:
        TN = TAB2.loc[s, f'TN_{mdl}_K']
        if mdl == 'Ising':
            d = RI[(RI.set == s) & (RI.L == 24) & (RI.B == 0)]
        else:
            d = pd.concat([R6[(R6.kind == 'H') & (R6.set == s)],
                           R7[(R7.kind == 'S') & (R7.set == s)]]).query('B==0')
        g = d.groupby('T').chi.mean()
        x = g.index.values / TN
        y = g.values * dJ[s]
        m = (x > 0.6) & (x < 1.35)
        CH += [dict(model=mdl, set=s, t=a, chi_norm=b) for a, b in zip(x[m], y[m])]

for n, (lbl, mk_) in {'Jena_yeni': ('Jena 2026', 'o'), 'Joy1992': ('Joy 1992', 's')}.items():
    T_, c_, nm = sets_exp[n]
    tf = FX.loc[FX.veri == n, 'TN_Fisher'].iloc[0]
    o = np.argsort(T_)
    x = T_[o] / tf
    y = c_[o] / nm
    m = (x > 0.6) & (x < 1.35)
    CH += [dict(model='exp', set=n, t=a, chi_norm=b) for a, b in zip(x[m], y[m])]

# Make figure
plt.close('all')
apply_figure_style()

fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
CHd = pd.DataFrame(CH)

for ax, mdl in zip(axs, ('Ising', 'Heis')):
    for s in rank7:
        d = CHd[(CHd.model == mdl) & (CHd.set == s)]
        ax.plot(d.t, d.chi_norm, '-', color=col[s], lw=1.0, label=short[s])
    for n, (lbl, mk_) in {'Jena_yeni': ('Jena 2026', 'o'), 'Joy1992': ('Joy 1992', 's')}.items():
        d = CHd[(CHd.model == 'exp') & (CHd.set == n)]
        ax.plot(d.t, d.chi_norm, mk_, mfc='white' if n == 'Joy1992' else 'k', mec='k', ms=3.2, ls='none', label=lbl, zorder=4)
    ax.axvspan(0.830, 1.103, color='0.9', lw=0, zorder=0)
    ax.axhline(1, color='k', ls=':', lw=0.8)
    ax.set_xlabel(r'$T/T_\mathrm{N}$')
    ax.set_xlim(0.6, 1.35)
    ax.set_title({'Ising': 'Blume–Capel (Ising limit)', 'Heis': 'classical Heisenberg + D'}[mdl], fontsize=7)

axs[0].set_ylabel(r'$\chi_\parallel\,[J(\mathbf{Q})-J(0)]$')
axs[0].set_ylim(0, 1.6)
axs[1].text(1.33, 1.03, r'mean field at $T_\mathrm{N}$', fontsize=6, ha='right', va='bottom')
axs[0].legend(loc='upper left', fontsize=6, ncol=1, frameon=False)
for ax, l in zip(axs, 'ab'):
    panel_letter(ax, l)

fig.tight_layout()
fig.savefig('Sekil_chi_normal.png', dpi=300)
fig.savefig('Sekil_chi_normal.pdf')