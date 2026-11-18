"""ozet5.py - mc5_raw.jsonl hizli ozet (login node): set basina kosu, B = 0 C tepesi T_N(C) (K), 5 T ve 6 T Maxwell Delta S_M tepe (J/kgK),
tepe T/T_N(C), 1 T dchi/dT en buyuk (muB/Fe/T/K). g = 2, M_mol 183.0 g/mol. Tam analiz yerelde. ASCII-only."""
import json
import numpy as np
GMU = 0.115768; CONV = 8.314462618 / 0.08617333262 / 0.183
TRAP = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
R = [json.loads(l) for l in open('mc5_raw.jsonl')]
print('%-15s %5s %7s %9s %9s %7s %10s' % ('set', 'kosu', 'TN_C', 'dSmax5T', 'dSmax6T', 'T/TN', 'dchidT'))
for n in sorted({r['set'] for r in R}):
    a = [r for r in R if r['set'] == n]; Ts = np.array(sorted({round(r['T'], 6) for r in a})); Bs = sorted({r['B'] for r in a})
    g = lambda b, k: [np.mean([r[k] for r in a if r['B'] == b and abs(r['T'] - T) < 1e-4] or [np.nan]) for T in Ts]
    M = np.array([g(b, 'm') for b in Bs])
    if np.isnan(M).any() or max(Bs) < 6 or len(Ts) < 3: print('%-15s %5d  eksik' % (n, len(a))); continue
    d = np.gradient(M, Ts, axis=1); h = GMU * np.array(Bs); C0 = np.array(g(0, 'C')); tnc = Ts[int(np.argmax(C0))]
    S = [TRAP(d[:i + 1], h[:i + 1], axis=0) * CONV for i in range(len(Bs))]; S5 = S[Bs.index(5)] if 5 in Bs else S[-1]; S6 = S[-1]
    dc = np.gradient(2 * M[Bs.index(1)], Ts).max()
    print('%-15s %5d %7.1f %9.4f %9.4f %7.3f %10.3e' % (n, len(a), tnc, S5.max(), S6.max(), Ts[int(np.argmax(S6))] / tnc, dc))
