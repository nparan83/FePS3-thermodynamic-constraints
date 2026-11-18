import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'analysis')); sys.path.insert(0, os.path.join(REPO, 'mc', 'lib'))
def P(f):
    for d in ('data/experiment', 'data/derived', 'mc/lib', 'analysis'):
        p = os.path.join(REPO, d, f)
        if os.path.exists(p): return p
    raise FileNotFoundError(f)

import os
import sys
import json
import tarfile
import hashlib
import numpy as np
import pandas as pd

from figstyle import apply_figure_style, panel_letter, META_GREY

GMU = 0.115768
CONV = 8.314462618 / 0.08617333262 / 0.183
WIN = {'ana': (0.830, 1.103), 'dar': (0.90, 1.05), 'genis': (0.75, 1.20)}

OD = os.path.join(REPO, 'mc')

# Load MF_kisitlar.csv
mf = pd.read_csv(P('MF_kisitlar.csv'))
dJ = dict(zip(mf.set, mf.dJ))

# Load MCE_5T_deney_karsilastirma.csv
Q = pd.read_csv(P('MCE_5T_deney_karsilastirma.csv'))

# Load mc_v7 tar
p7 = os.path.join(OD, 'mc_v7', 'mc_v7_sonuc.tar.gz')
os.makedirs('v7', exist_ok=True)
with tarfile.open(p7) as t:
    t.extractall('v7')
R7 = pd.DataFrame([json.loads(l) for l in open('v7/mc7_raw.jsonl')])

# Load mc_v6 tar
p6 = os.path.join(OD, 'mc_v6', 'mc_v6_sonuc.tar.gz')
os.makedirs('v6', exist_ok=True)
with tarfile.open(p6) as t:
    names = [m.name for m in t.getmembers()]
    t.extractall('v6')
f6 = [n for n in names if n.endswith('.jsonl')]
R6 = pd.concat([pd.DataFrame([json.loads(l) for l in open('v6/' + n)]).assign(src=n) for n in f6])

# Load mc_v3/v4/v5 Ising data
RI = []
for v in ('3', '4', '5'):
    with tarfile.open(os.path.join(OD, f'mc_v{v}', f'mc_v{v}_sonuc.tar.gz')) as t:
        for m in t.getmembers():
            if m.name.endswith('.jsonl'):
                d = pd.DataFrame([json.loads(l) for l in t.extractfile(m).read().decode().splitlines() if l.strip()])
                d['src'] = 'v' + v
                RI.append(d)
RI = pd.concat(RI, ignore_index=True)
if 'L' not in RI:
    RI['L'] = 24
RI['L'] = RI['L'].fillna(24)

# ana7.py functions
def _tab(d, col):
    x = d.groupby('T')[col].mean()
    return x.index.values, x.values

def _at(d, col, T):
    t, y = _tab(d, col)
    if T < t.min() - 1e-9 or T > t.max() + 1e-9:
        return np.nan
    return float(np.interp(T, t, y))

def _one(d, TN, dJ_val, t1, t2, Bmax=5.0):
    d0 = d[d.B == 0]
    F0 = dJ_val * (_at(d0, 'chi', t2 * TN) - _at(d0, 'chi', t1 * TN))
    Bs = sorted(b for b in d.B.unique() if b <= Bmax + 1e-9)
    if abs(Bs[-1] - Bmax) > 1e-9:
        return F0, np.nan, np.nan
    dm = [_at(d[d.B == b], 'm', t2 * TN) - _at(d[d.B == b], 'm', t1 * TN) for b in Bs]
    I = np.trapezoid(dm, GMU * np.array(Bs))
    h5 = GMU * Bmax
    return F0, dJ_val * 2 * I / h5**2, I * CONV / (t2 - t1)

def case(d, TN, dJ_val, sTN=0.0, seedcol='seed'):
    out = {}
    seeds = sorted(d[seedcol].dropna().unique()) if seedcol in d and d[seedcol].notna().any() else [None]
    for w, (t1, t2) in WIN.items():
        v = np.array(_one(d, TN, dJ_val, t1, t2))
        ps = np.array([_one(d[d[seedcol] == s], TN, dJ_val, t1, t2) for s in seeds]) if len(seeds) > 1 else None
        se = ps.std(0, ddof=1) / np.sqrt(len(seeds)) if ps is not None else np.full(3, np.nan)
        lo = np.array(_one(d, TN - sTN, dJ_val, t1, t2))
        hi = np.array(_one(d, TN + sTN, dJ_val, t1, t2))
        for k, nm in enumerate(('F0', 'F5', 'prod')):
            out[f'{nm}_{w}'] = v[k]
            out[f'{nm}_{w}_se'] = se[k]
            out[f'{nm}_{w}_sTN'] = 0.5 * abs(hi[k] - lo[k])
    out['nseed'] = len(seeds)
    return out

def pk(T, y, npt=2):
    T = np.asarray(T)
    y = np.asarray(y)
    j = int(np.nanargmax(y))
    lo, hi = max(j - npt, 0), min(j + npt + 1, len(T))
    if hi - lo < 3:
        return T[j]
    c = np.polyfit(T[lo:hi], y[lo:hi], 2)
    t = -c[1] / (2 * c[0]) if c[0] < 0 else T[j]
    return t if T[lo] <= t <= T[hi - 1] else T[j]

def tn_peak(d, col='C'):
    t, y = _tab(d[d.B == 0], col)
    return pk(t, y)

# Compute PK from B7 (Binder block)
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
        r[f'edge{L}'] = int(np.argmax(d['C'].values)) in (0, len(d) - 1)
    rows_pk.append(r)
PK = pd.DataFrame(rows_pk)
PK['dC_36_48%'] = 100 * (PK.C48 - PK.C36) / PK.C48
PK['dX_36_48%'] = 100 * (PK.X48 - PK.X36) / PK.X48
PK['C_X_48%'] = 100 * (PK.C48 - PK.X48) / PK.C48
PK['sTN'] = PK.apply(lambda r: 0.5 * (max(r.C36, r.C48, r.X36, r.X48) - min(r.C36, r.C48, r.X36, r.X48)), axis=1)

pkd = {(r.model, r.set): r for r in PK.itertuples()}

# published model mapping
base = {
    'W4pub': 'Wildes20_r4_x2',
    'W2pub': 'Wildes20_r2_x2',
    'OlsenD': 'Olsen_U2',
    'YangD': 'Yang_U36',
    'AmirDB': 'Amir_U222'
}

P7 = R7[R7.kind == 'P']
Z7 = R7[R7.kind == 'Z']
S7 = R7[R7.kind == 'S']

rank7 = ['Wildes20_r4_x2', 'Wildes20_r2_x2', 'Lancon_x2', 'Yang_U36', 'Amir_U222', 'Olsen_U2', 'Chat_U25']

rows = []
for s in dJ:
    if s.startswith('experiment'):
        continue
    # Ising
    d = RI[(RI.set == s) & (RI.L == 24)]
    r = pkd[('I', s)]
    rows.append(dict(kind='2D', model='Ising', set=s, TN=r.C48, sTN=r.sTN, TN24=r.C24,
                     **case(d, r.C48, dJ[s], r.sTN)))
    # Heis
    d = pd.concat([R6[(R6.kind == 'H') & (R6.set == s)], R7[(R7.kind == 'S') & (R7.set == s)]])
    r = pkd[('H', s)]
    rows.append(dict(kind='2D', model='Heis', set=s, TN=r.C48, sTN=r.sTN, TN24=r.C24,
                     **case(d, r.C48, dJ[s], r.sTN)))

for m, b in base.items():
    d = P7[P7.set == m]
    TN = tn_peak(d)
    r = pkd[('H', b)]
    sT = np.hypot(r.sTN, r.C24 - r.C48)
    rows.append(dict(kind='pub', model='Heis+pub', set=m, base=b, D=d.D.iloc[0], TN=TN, sTN=sT, TN24=TN,
                     **case(d, TN, dJ[b], sT)))

# 3D
D3 = pd.concat([R6[R6.kind == '3'].assign(src='v6'), Z7.assign(src='v7')])
D3['Jp'] = D3.Jp.round(4)
for (s, jp), d in D3.groupby(['set', 'Jp']):
    TN = tn_peak(d)
    rows.append(dict(kind='3D', model=f'Ising3D Jp{jp}', set=s, TN=TN, sTN=np.nan, TN24=TN,
                     TN2D24=pkd[('I', s)].C24,
                     **case(d.assign(seed=d.src), TN, dJ[s], 0.0)))

T3 = pd.DataFrame(rows)
T3.to_csv('MC_v7_durumlar.csv', index=False)
print(f"MC_v7_durumlar.csv saved with {len(T3)} rows")