"""mf_constraints.py - mean-field (RPA) quantities of Sec. II C and III B for every parameter set.
Columns reproduced: set, dJ = J(Q) - J(0) [meV], dE_eff_meV = E_FM - E_zz = S^2 dJ / 2, frus_J0_over_JQ = J(0)/J(Q),
TN_BC / TN_cl = mean-field T_N of the S = 2 Blume-Capel (Ising-limit) and classical |S| = 2 models with D = 2.66 meV [K],
F_BC / F_cl = dJ * [chi(t2 T_N) - chi(t1 T_N)] in mean field, window 0.830-1.103.
Convention: H = -sum_bonds J S_i.S_j - D sum (S_i^z)^2, S = 2, J > 0 ferromagnetic. J(Q), J(0) from mc/lib/fepslib2.py.
Output: MF_kisitlar_regen.csv (compare with data/derived/MF_kisitlar.csv)."""
import os, sys
import numpy as np, pandas as pd
from scipy.optimize import brentq
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'mc', 'lib'))
import fepslib2 as g2
KB = 0.08617333262; S = 2.0; D = 2.66; T1, T2 = 0.830, 1.103
SETS = {'Wildes20_r4_x2': (1.96, 1.96, -0.31, -0.90, 0.0), 'Wildes20_r2_x2': (1.80, 1.80, -0.094, -1.28, 0.0),
        'Lancon_x2': (2.92, 2.92, -0.08, -1.92, 0.0), 'Ilyas24_x2': (1.12, 2.80, -0.02, -1.06, 0.0, -0.86),
        'Yang_U36': (3.13, 3.13, -0.34, -2.01, 0.0), 'Amir_U222': (1.0025, -0.815, -0.31, -1.4275, 0.375),
        'Olsen_U2': (2.10, 2.10, -0.21, -2.60, 0.0), 'Chat_U25': (2.60, -2.20, -1.90, -0.60, 0.0)}
L = 24; PAT = [p for p in g2.zigzag_patterns(L, L) if p is not None][0].astype(float)
def jq0(J):
    fq = (g2.local_field(PAT, J) * PAT).mean(); f0 = g2.local_field(np.ones((L, L)), J).mean(); return fq, f0
MS = np.arange(-2, 3, dtype=float); X = np.linspace(-1, 1, 4001)
def site(T, h, model):
    """single-site moments <Sz>, <Sz^2> in field h (meV) at T (K)."""
    b = 1 / (KB * T)
    if model == 'BC': s = MS
    else: s = S * X
    w = np.exp(b * (D * s ** 2 + h * s) - np.max(b * (D * s ** 2 + h * s)))
    return (w * s).sum() / w.sum(), (w * s ** 2).sum() / w.sum()
def tn(JQ, model):
    return brentq(lambda T: JQ * site(T, 0.0, model)[1] / (KB * T) - 1, 1.0, 2000.0)
def chi(T, JQ, J0, TN, model):
    if T >= TN: m = 0.0
    else:
        m = brentq(lambda x: site(T, JQ * x, model)[0] - x, 1e-6, S)
    s1, s2 = site(T, JQ * m, model); cl = (s2 - s1 ** 2) / (KB * T)
    return cl / (1 - J0 * cl)
rows = []
for k, J in SETS.items():
    JQ, J0 = jq0(J); dJ = JQ - J0; r = dict(set=k, dJ=dJ, dE_eff_meV=S ** 2 * dJ / 2, frus_J0_over_JQ=J0 / JQ)
    for model, tag in (('BC', 'BC'), ('cl', 'cl')):
        TN = tn(JQ, model); r['TN_' + tag] = TN
        r['F_' + tag] = dJ * (chi(T2 * TN, JQ, J0, TN, model) - chi(T1 * TN, JQ, J0, TN, model))
    rows.append(r)
pd.DataFrame(rows).round(4).to_csv('MF_kisitlar_regen.csv', index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False))
