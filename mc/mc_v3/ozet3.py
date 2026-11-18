"""ozet3.py - mc3_raw.jsonl hizli ozet (login node): set basina kosu sayisi, 6 T Maxwell Delta S_M tepe (J/kgK), tepe T/T_N, M(6 T) en buyuk.
Tam analiz (n, RCP, chi, sekiller) yerelde yapilir. g = 2, M_mol 183.0 g/mol. ASCII-only."""
import json
import numpy as np
GMU = 0.115768; CONV = 8.314462618 / 0.08617333262 / 0.183
R = [json.loads(l) for l in open('mc3_raw.jsonl')]
print('%-10s %5s %10s %7s %9s' % ('set', 'kosu', 'dSmax6T', 'T/TN', 'M6T_max'))
for n in sorted({r['set'] for r in R}):
    a = [r for r in R if r['set'] == n]; tn = a[0]['TN']; Ts = np.array(sorted({round(r['T'], 6) for r in a})); Bs = sorted({r['B'] for r in a})
    M = np.array([[np.mean([r['m'] for r in a if r['B'] == b and abs(r['T'] - T) < 1e-4] or [np.nan]) for T in Ts] for b in Bs])
    if np.isnan(M).any() or max(Bs) < 6: print('%-10s %5d  eksik' % (n, len(a))); continue
    d = np.gradient(M, Ts, axis=1); h = GMU * np.array(Bs)
    S = getattr(np, 'trapezoid', getattr(np, 'trapz', None))(d, h, axis=0) * CONV; k = int(np.argmax(S))
    print('%-10s %5d %10.4f %7.3f %9.4f' % (n, len(a), S[k], Ts[k] / tn, 2 * M[-1].max()))
