"""ana7.py - Makale 1 adim 5: MC v7 + v6 + v3-v5 Ising verisinden uc kisit tablosu.
Tanimlar (spin birimi, S=2, h = g muB B ile g=2; GMU = 0.115768 meV/T):
  T_N        : C(T) tepesi (parabol), Ising/Heis D2.66 icin L48 (B blogu); yayimlanmis modeller icin L24 (P blogu).
  sigma_TN   : {C36, C48, chi_zz36, chi_zz48} tepe konumlarinin yari-araligi (B blogu); P icin taban setin degeri.
  F0         : dJ * [chi(t2 T_N) - chi(t1 T_N)], chi = B=0 dalgalanma duygunlugu (S_z/meV/spin); dJ = J(Q)-J(0) (MF_kisitlar.csv).
  F5         : dJ * 2*int_0^h5 [m(t2 T_N,h)-m(t1 T_N,h)] dh / h5^2  (5 T, tam ozdeslik, lineer cevapta F0'a esit).
  prod       : <dS_M>_pencere * T_N = CONV * int_0^h5 dm dh / (t2-t1)  [J/kg, g=2; g ile ~g^2 olcekli].
  hata       : tohumlar arasi standart hata (n tohum) ve T_N +- sigma ile yeniden hesap (ayri sutun).
"""
import numpy as np, pandas as pd
GMU = 0.115768; CONV = 8.314462618 / 0.08617333262 / 0.183
WIN = {'ana': (0.830, 1.103), 'dar': (0.90, 1.05), 'genis': (0.75, 1.20)}

def _tab(d, col):
    x = d.groupby('T')[col].mean(); return x.index.values, x.values

def _at(d, col, T):
    t, y = _tab(d, col)
    if T < t.min() - 1e-9 or T > t.max() + 1e-9: return np.nan
    return float(np.interp(T, t, y))

def _one(d, TN, dJ, t1, t2, Bmax=5.0):
    """tek tohum (veya birlesik) icin F0, F5, prod."""
    d0 = d[d.B == 0]
    F0 = dJ * (_at(d0, 'chi', t2 * TN) - _at(d0, 'chi', t1 * TN))
    Bs = sorted(b for b in d.B.unique() if b <= Bmax + 1e-9)
    if abs(Bs[-1] - Bmax) > 1e-9: return F0, np.nan, np.nan
    dm = [_at(d[d.B == b], 'm', t2 * TN) - _at(d[d.B == b], 'm', t1 * TN) for b in Bs]
    I = np.trapezoid(dm, GMU * np.array(Bs)); h5 = GMU * Bmax
    return F0, dJ * 2 * I / h5**2, I * CONV / (t2 - t1)

def case(d, TN, dJ, sTN=0.0, seedcol='seed'):
    """d: tek set/model; tum B ve tohumlar. Donus: pencere basina deger + tohum hatasi + T_N duyarliligi."""
    out = {}
    seeds = sorted(d[seedcol].dropna().unique()) if seedcol in d and d[seedcol].notna().any() else [None]
    for w, (t1, t2) in WIN.items():
        v = np.array(_one(d, TN, dJ, t1, t2))
        ps = np.array([_one(d[d[seedcol] == s], TN, dJ, t1, t2) for s in seeds]) if len(seeds) > 1 else None
        se = ps.std(0, ddof=1) / np.sqrt(len(seeds)) if ps is not None else np.full(3, np.nan)
        lo = np.array(_one(d, TN - sTN, dJ, t1, t2)); hi = np.array(_one(d, TN + sTN, dJ, t1, t2))
        for k, nm in enumerate(('F0', 'F5', 'prod')):
            out[f'{nm}_{w}'] = v[k]; out[f'{nm}_{w}_se'] = se[k]; out[f'{nm}_{w}_sTN'] = 0.5 * abs(hi[k] - lo[k])
    out['nseed'] = len(seeds); return out

def pk(T, y, npt=2):
    T = np.asarray(T); y = np.asarray(y); j = int(np.nanargmax(y)); lo, hi = max(j - npt, 0), min(j + npt + 1, len(T))
    if hi - lo < 3: return T[j]
    c = np.polyfit(T[lo:hi], y[lo:hi], 2); t = -c[1] / (2 * c[0]) if c[0] < 0 else T[j]
    return t if T[lo] <= t <= T[hi - 1] else T[j]

def tn_peak(d, col='C'):
    t, y = _tab(d[d.B == 0], col); return pk(t, y)
