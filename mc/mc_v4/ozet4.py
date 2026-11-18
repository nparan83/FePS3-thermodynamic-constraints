"""ozet4.py - mc4_raw.jsonl hizli ozet (login node): (set, L) basina kosu sayisi, 6 T Maxwell Delta S_M tepe (J/kgK), tepe T/T_N,
B = 0 C tepesi T/T_N (kaba), 1 T dchi/dT en buyuk (muB/Fe/T/K). g = 2, M_mol 183.0 g/mol. Tam analiz yerelde. ASCII-only."""
import json
import numpy as np
GMU = 0.115768; CONV = 8.314462618 / 0.08617333262 / 0.183
TRAP = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
R = [json.loads(l) for l in open('mc4_raw.jsonl')]
print('%-10s %3s %5s %9s %7s %7s %10s' % ('set', 'L', 'kosu', 'dSmax6T', 'T/TN', 'TC/TN', 'dchidT'))
for n, L in sorted({(r['set'], r['L']) for r in R}):
    a = [r for r in R if r['set'] == n and r['L'] == L]; tn = a[0]['TN']; Ts = np.array(sorted({round(r['T'], 6) for r in a})); Bs = sorted({r['B'] for r in a})
    g = lambda b, k: [np.mean([r[k] for r in a if r['B'] == b and abs(r['T'] - T) < 1e-4] or [np.nan]) for T in Ts]
    M = np.array([g(b, 'm') for b in Bs])
    if np.isnan(M).any() or max(Bs) < 6 or len(Ts) < 3: print('%-10s %3d %5d  eksik' % (n, L, len(a))); continue
    d = np.gradient(M, Ts, axis=1); S = TRAP(d, GMU * np.array(Bs), axis=0) * CONV; k = int(np.argmax(S))
    C0 = np.array(g(0, 'C')); dc = np.gradient(2 * M[Bs.index(1)] if 1 in Bs else M[1], Ts).max()
    print('%-10s %3d %5d %9.4f %7.3f %7.3f %10.3e' % (n, L, len(a), S[k], Ts[k] / tn, Ts[int(np.argmax(C0))] / tn, dc))
