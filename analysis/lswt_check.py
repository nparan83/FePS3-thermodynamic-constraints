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
import sys
import json
import numpy as np
import pandas as pd

from figstyle import apply_figure_style, panel_letter, META_GREY

import importlib, types

import fepslib2 as g2  # mc/lib/fepslib2.py (MC v5 library)


GMU = 0.115768

# Exchange sets
J6 = {
    'Wildes20_r4_x2': (1.96, 1.96, -0.31, -0.90, 0.0),
    'Wildes20_r2_x2': (1.80, 1.80, -0.094, -1.28, 0.0),
    'Ilyas24_x2': (1.12, 2.80, -0.02, -1.06, 0.0, -0.86),
    'Lancon_x2': (2.92, 2.92, -0.08, -1.92, 0.0),
    'Yang_U36': (3.13, 3.13, -0.34, -2.01, 0.0),
    'Olsen_U2': (2.10, 2.10, -0.21, -2.60, 0.0),
    'Amir_U222': (1.0025, -0.815, -0.31, -1.4275, 0.375),
    'own_U3.50': (0.7479, 0.7479, -0.0026, -2.7487, 0.0),
    'own_U3.00': (0.9767, 0.9767, -0.0053, -3.1433, 0.0),
    'own_U2.25': (1.5261, 1.5261, 0.0013, -3.9063, 0.0),
    'Chat_U25': (2.60, -2.20, -1.90, -0.60, 0.0),
}

NB = g2.NB

# Determine zigzag pattern
L = 24
pats = [p for p in g2.zigzag_patterns(L, L) if p is not None]
pat = pats[0].astype(int)

def period(P):
    Lz = P.shape[0]
    for px in range(1, Lz + 1):
        if Lz % px == 0 and np.array_equal(np.roll(P, px, 0), P):
            break
    for py in range(1, Lz + 1):
        if Lz % py == 0 and np.array_equal(np.roll(P, py, 1), P):
            break
    return px, py

px, py = period(pat)
px = max(px, 2) if px % 2 else px
py = py if (py % 2 == 0) else 2 * py

def lswt(J, D, S=2.0, nk=48, quantum=False):
    Jd = g2.jvec(J)
    sites = [(i, j) for i in range(px) for j in range(py)]
    idx = {s: n for n, s in enumerate(sites)}
    n = len(sites)
    sig = np.array([pat[i, j] for i, j in sites], float)
    A0 = np.zeros(n)
    bonds = []
    for a, (i, j) in enumerate(sites):
        p = (i + j) % 2
        for di, dj, lab in NB[p]:
            Jv = Jd.get(lab, 0.0)
            if Jv == 0:
                continue
            ii, jj = i + di, j + dj
            b = idx[(ii % px, jj % py)]
            R = (ii // px, jj // py)
            sgn = sig[a] * sig[b]
            A0[a] += S * Jv * sgn
            bonds.append((a, b, Jv, sgn, R))
    A0 += (2 * S - 1) * D if quantum else 2 * S * D
    ks = [(k1, k2) for k1 in np.linspace(0, 2 * np.pi, nk, endpoint=False)
          for k2 in np.linspace(0, 2 * np.pi, nk, endpoint=False)]
    W = []
    for k1, k2 in ks:
        Ak = np.diag(A0).astype(complex)
        Bk = np.zeros((n, n), complex)
        for a, b, Jv, sgn, R in bonds:
            ph = np.exp(1j * (k1 * R[0] + k2 * R[1]))
            if sgn > 0:
                Ak[a, b] += -Jv * S * ph
            else:
                Bk[a, b] += -Jv * S * ph
        Ak = (Ak + Ak.conj().T) / 2
        Bk = (Bk + Bk.T) / 2
        M = np.block([[Ak, Bk], [Bk.conj(), Ak.conj()]])
        eta = np.diag(np.r_[np.ones(n), -np.ones(n)])
        ev = np.linalg.eigvals(eta @ M)
        W.append(np.sort(ev.real)[n:])
    W = np.array(W)
    return W.min(), W.max(), np.isfinite(W).all()

tests = {
    'Lancon x1': ((1.46, 1.46, -0.04, -0.96, 0.0), 2.66),
    'Lancon x2': ((2.92, 2.92, -0.08, -1.92, 0.0), 2.66),
    'Wildes r4 x1': ((0.98, 0.98, -0.155, -0.45, 0.0), 2.52),
    'Wildes r4 x2': ((1.96, 1.96, -0.31, -0.90, 0.0), 2.52),
    'Wildes r2 x1': ((0.90, 0.90, -0.047, -0.64, 0.0), 2.53),
    'Wildes r2 x2': ((1.80, 1.80, -0.094, -1.28, 0.0), 2.53),
}

lw = []
for k, (J, D) in tests.items():
    for q in (False, True):
        mn, mx, _ = lswt(J, D, nk=36, quantum=q)
        lw.append(dict(set=k, D=D,
                       anisotropy_term='(2S-1)D quantum' if q else '2SD classical',
                       gap=mn, top=mx))

LW = pd.DataFrame(lw)
LW['exp_gap'] = '15-17 meV (Lancon 2016, (0k0) cut; powder ~16)'
LW['exp_top'] = '~40 meV powder band (Wildes 2012 via Lancon 2016); ~25 meV along (0k0)'
LW.round(2).to_csv('LSWT_kontrol.csv', index=False)
print(LW[['set', 'anisotropy_term', 'gap', 'top']].round(1).to_string())