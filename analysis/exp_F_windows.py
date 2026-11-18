"""exp_F_windows.py - experimental F = chi_par(t2 T_N) - chi_par(t1 T_N), normalized by the mean-field chi, for three T/T_N windows and three digitized datasets (Jena 2026 old/new digitization, Joy 1992); T_N from max d(chi)/dT and from Fisher's relation, with bootstrap SDs. Writes F_deney_pencereler.csv (SM Table S4)."""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'analysis')); sys.path.insert(0, os.path.join(REPO, 'mc', 'lib'))
def P(f):
    for d in ('data/experiment', 'data/derived', 'mc/lib', 'analysis'):
        p = os.path.join(REPO, d, f)
        if os.path.exists(p): return p
    raise FileNotFoundError(f)

import os
import re
import json
import glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from figstyle import apply_figure_style, panel_letter, META_GREY

joy = pd.read_csv(P('Joy_1992_fig3_X_m_vs_K.csv'), sep=';', decimal=',', header=None, names=['T', 'y'], encoding='utf-8-sig')
ja = pd.read_csv(P('new_Fig3a_Data_0.05T_Dik.csv'), sep=';', decimal=',', header=None, names=['T', 'y'], encoding='utf-8-sig')

od0 = os.path.join(REPO, 'mc')
fa = pd.read_csv(P('Fig3a_Data_0.05T_Dik.csv'), sep=';', decimal=',', header=None, names=['T', 'M'])
fa_clean = fa[fa['T'] > 0]

# Constants
GMU = 0.115768
CONV = 8.314462618 / 0.08617333262 / 0.183
WIN = {'ana': (0.830, 1.103), 'dar': (0.90, 1.05), 'genis': (0.75, 1.20)}

# chiMF = Msat^2 / (2 * dE)
# Msat = 4 muB/Fe in emu units per mol
chiMF = 122.1**2 / (2 * 4924.0)
chiMF_mol = chiMF * 1e-4 * 183.0

dEexp = 9.3367

def Fgen(T, chi, TN, t1, t2, norm):
    o = np.argsort(T)
    T = np.asarray(T)[o]
    chi = np.asarray(chi)[o]
    return (np.interp(t2 * TN, T, chi) - np.interp(t1 * TN, T, chi)) / norm

def tn_dchi(T, chi):
    g = pd.Series(chi, index=T).groupby(level=0).mean().sort_index()
    t = g.index.values
    y = g.values
    tt = np.linspace(t.min(), t.max(), 4000)
    yy = np.interp(tt, t, y)
    dy = np.gradient(yy, tt)
    m = (tt > 100) & (tt < 135)
    return tt[m][np.argmax(dy[m])]

sets_exp = {
    'Jena_eski': (fa_clean['T'].values, fa_clean.M.values / 0.05, chiMF),
    'Jena_yeni': (ja['T'].values, ja.y.values / 0.05, chiMF),
    'Joy1992': (joy['T'].values, joy.y.values, chiMF_mol)
}

rng = np.random.default_rng(1)

rows = []
for n, (T_, c_, nm) in sets_exp.items():
    r = dict(veri=n, TN_dchidT_max=tn_dchi(T_, c_))
    for w, (t1, t2) in WIN.items():
        for TN in (118, 119, 121, 123, 125):
            r[f'{w}_TN{TN}'] = Fgen(T_, c_, TN, t1, t2, nm)
        bsv = []
        for _ in range(1000):
            Tj = T_ + rng.normal(0, 0.5, len(T_))
            cj = c_ * (1 + rng.normal(0, 0.01, len(c_)))
            bsv.append(Fgen(Tj, cj, 119, t1, t2, nm))
        r[f'{w}_bs_sd'] = np.std(bsv)
    r['chi_peak_emu_mol_Oe'] = (c_.max() / 1e4 * 183.0) if n != 'Joy1992' else c_.max()
    rows.append(r)

FX = pd.DataFrame(rows)

def tn_fisher(T, chi, lo=100, hi=135):
    g = pd.Series(chi, index=T).groupby(level=0).mean().sort_index()
    t = g.index.values
    y = g.values * t
    tt = np.linspace(lo, hi, 3501)
    yy = np.interp(tt, t, y)
    dy = np.gradient(yy, tt)
    return tt[np.argmax(dy)]

for n, (T_, c_, nm) in sets_exp.items():
    tf = tn_fisher(T_, c_)
    bsf = [tn_fisher(T_ + rng.normal(0, 0.5, len(T_)), c_ * (1 + rng.normal(0, 0.01, len(c_)))) for _ in range(500)]
    FX.loc[FX.veri == n, 'TN_Fisher'] = tf
    FX.loc[FX.veri == n, 'TN_Fisher_bs_sd'] = np.std(bsf)
    for w, (t1, t2) in WIN.items():
        FX.loc[FX.veri == n, f'{w}_TNFisher'] = Fgen(T_, c_, tf, t1, t2, nm)

FX.to_csv('F_deney_pencereler.csv', index=False)