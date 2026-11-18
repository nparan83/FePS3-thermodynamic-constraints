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
import itertools
import numpy as np
import pandas as pd

import fepslib2 as _g2  # mc/lib/fepslib2.py (MC v5 library: J2par/J2perp)
g2 = vars(_g2)

NB = g2['NB']
jvec = g2['jvec']

GMU = 0.115768

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

# Build zigzag pattern
L = 24
pats = [p for p in g2['zigzag_patterns'](L, L) if p is not None]
pat = pats[0]

# Get labels
labels = sorted({lab for p in (0, 1) for _, _, lab in NB[p]})


def cell_feats(cx, cy):
    n = cx * cy
    idxs = [(i, j) for i in range(cx) for j in range(cy)]
    pairs = {l: [] for l in labels}
    for a, (i, j) in enumerate(idxs):
        p = (i + j) % 2
        for di, dj, lab in NB[p]:
            b = ((i + di) % cx) * cy + ((j + dj) % cy)
            pairs[lab].append((a, b))
    confs = np.array(list(itertools.product([-1, 1], repeat=n)), dtype=np.int8)
    F = np.stack([
        0.5 * sum(confs[:, a] * confs[:, b] for a, b in pairs[l]).astype(float) / n
        if pairs[l] else np.zeros(len(confs))
        for l in labels
    ], 1)
    m = confs.mean(1)
    U = np.unique(np.round(np.c_[F, m], 6), axis=0)
    return U


cells = [(2, 2), (2, 4), (4, 2), (4, 4), (2, 6), (6, 2), (2, 8), (8, 2)]
UU = np.unique(np.vstack([cell_feats(*c) for c in cells]), axis=0)


def jlab(J):
    d = jvec(J)
    return np.array([d.get(l, 0.0) for l in labels])


def spinflip(J, S=2, Bmax=150):
    jl = jlab(J)
    E0 = -S ** 2 * (UU[:, :-1] @ jl)
    m = UU[:, -1]
    Bs = np.linspace(0, Bmax, 6001)
    E = E0[None, :] - GMU * Bs[:, None] * S * m[None, :]
    g_ = E.argmin(1)
    mg = m[g_]
    tr = [(round(Bs[i], 2), round(mg[i - 1], 3), round(mg[i], 3))
          for i in range(1, len(Bs)) if abs(mg[i] - mg[i - 1]) > 1e-6]
    return tr, mg[0]


# Compute dJ for each set (JQ - J0 from mean field)
def JQ0(J):
    P = pat.astype(float)
    fz = g2['local_field'](P, J)
    fq = (fz * P).mean()
    f0 = g2['local_field'](np.ones((L, L)), J).mean()
    return fq, f0


# LSWT computation
px_cell, py_cell = 2, 4  # magnetic cell for zigzag


def lswt(J, D, S=2.0, nk=36, quantum=False):
    n = px_cell * py_cell
    sites = [(i, j) for i in range(px_cell) for j in range(py_cell)]
    idx = {s: k for k, s in enumerate(sites)}
    P_cell = pat[:px_cell, :py_cell]
    sig = np.array([P_cell[i, j] for i, j in sites], float)

    A0 = np.zeros(n)
    bonds = []
    for a, (i, j) in enumerate(sites):
        p = (i + j) % 2
        for di, dj, lab in NB[p]:
            Jv = jvec(J).get(lab, 0.0)
            if Jv == 0:
                continue
            ii, jj = i + di, j + dj
            b = idx[(ii % px_cell, jj % py_cell)]
            R = (ii // px_cell, jj // py_cell)
            sgn = sig[a] * sig[b]
            A0[a] += S * Jv * sgn
            bonds.append((a, b, Jv, sgn, R))
    A0 += (2 * S - 1) * D if quantum else 2 * S * D

    ks = [(k1, k2)
          for k1 in np.linspace(0, 2 * np.pi, nk, endpoint=False)
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


rows = []
for k, J in J6.items():
    JQ, J0 = JQ0(J)
    dJ = JQ - J0
    tr, _ = spinflip(J)
    mn, mx, _ = lswt(J, 2.66, nk=36)
    rows.append(dict(
        set=k,
        dE_eff_meV=round(2 * dJ, 6),
        B_direct_T=round(2 * dJ / (GMU * 2), 6),
        first_transition_T=tr[0][0] if tr else np.nan,
        plateau_half=any(abs(t[2] - 0.5) < 1e-3 for t in tr),
        transitions=str([(float(a), float(b), float(c)) for a, b, c in tr]),
        lswt_gap=mn,
        lswt_top=mx,
    ))

SFd = pd.DataFrame(rows)
SFd.round(3).to_csv('spinflip_alanlari.csv', index=False)