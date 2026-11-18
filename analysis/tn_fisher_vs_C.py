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

from figstyle import apply_figure_style, panel_letter, META_GREY

# Load MC v7 data
import tarfile
import json
import os

od0 = os.path.join(REPO, 'mc')

# Load MC v7 Binder data
p7 = os.path.join(od0, 'mc_v7', 'mc_v7_sonuc.tar.gz')
os.makedirs('v7', exist_ok=True)
with tarfile.open(p7) as t:
    t.extractall('v7')
R7 = pd.DataFrame([json.loads(l) for l in open('v7/mc7_raw.jsonl')])

B7 = R7[R7.kind == 'B']

def pkfit(T, y, npt=2):
    T = np.asarray(T); y = np.asarray(y); j = int(np.nanargmax(y))
    lo, hi = max(j - npt, 0), min(j + npt + 1, len(T))
    if hi - lo < 3:
        return T[j]
    pc = np.polyfit(T[lo:hi], y[lo:hi], 2)
    t = -pc[1] / (2 * pc[0]) if pc[0] < 0 else T[j]
    return t if T[lo] <= t <= T[hi - 1] else T[j]

rows = []
for (mdl, s), g in B7.groupby(['model', 'set']):
    r = dict(model=mdl, set=s)
    for L in (24, 36, 48):
        d = g[g.L == L].sort_values('T')
        r[f'C{L}'] = pkfit(d['T'], d['C'])
        r[f'X{L}'] = pkfit(d['T'], d['chizz'])
        r[f'edge{L}'] = int(np.argmax(d['C'].values)) in (0, len(d) - 1)
    rows.append(r)
PK = pd.DataFrame(rows)
PK['dC_36_48%'] = 100 * (PK.C48 - PK.C36) / PK.C48
PK['dX_36_48%'] = 100 * (PK.X48 - PK.X36) / PK.X48
PK['C_X_48%'] = 100 * (PK.C48 - PK.X48) / PK.C48

pkd = {(r.model, r.set): r for r in PK.itertuples()}

# Compute Fisher T_N from MC data
def tn_fisher(T, chi, lo=100, hi=135):
    g = pd.Series(chi, index=T).groupby(level=0).mean().sort_index()
    t = g.index.values; y = g.values * t
    tt = np.linspace(lo, hi, 3501); yy = np.interp(tt, t, y); dy = np.gradient(yy, tt)
    return tt[np.argmax(dy)]

chk = []
for (mdl, s), g in B7[B7.L == 48].groupby(['model', 'set']):
    d = g.sort_values('T'); t = d['T'].values; y = d['chi'].values * t
    tt = np.linspace(t.min(), t.max(), 2000); dy = np.gradient(np.interp(tt, t, y), tt)
    chk.append(dict(model=mdl, set=s, TN_C48=pkd[(mdl, s)].C48, TN_Fisher48=tt[np.argmax(dy)]))

CK = pd.DataFrame(chk)
CK['fark_%'] = 100 * (CK.TN_Fisher48 - CK.TN_C48) / CK.TN_C48

CK.to_csv('TN_Fisher_vs_C_MC.csv', index=False)