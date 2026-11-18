"""ozet7.py - mc7_raw.jsonl hizli ozet (login node). P/Z/S: T_N (B = 0 C tepesi, parabol), 5 T pencere (0.830-1.103 T_N) Maxwell <dS> ve carpim,
tohum farki; B: Binder U4 kesisimleri (L 24-36, 36-48). g = 2, M_mol 183.0. ASCII-only."""
import json
import numpy as np
GMU = 0.115768; CONV = 8.314462618 / 0.08617333262 / 0.183; TRAP = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
R = [json.loads(l) for l in open('mc7_raw.jsonl')]
def ana(a):
    Ts = np.array(sorted({round(r['T'], 4) for r in a})); Bs = sorted({r['B'] for r in a})
    g = lambda b, k: np.array([np.mean([r[k] for r in a if r['B'] == b and abs(r['T'] - T) < 1e-3] or [np.nan]) for T in Ts])
    M = np.array([g(b, 'm') for b in Bs]); C0 = g(0, 'C'); j = int(np.nanargmax(C0)); jj = slice(max(j - 2, 0), j + 3)
    pc = np.polyfit(Ts[jj], C0[jj], 2); tn = -pc[1] / (2 * pc[0]); i5 = Bs.index(5)
    S5 = TRAP(np.gradient(M[:i5 + 1], Ts, axis=1), GMU * np.array(Bs[:i5 + 1]), axis=0) * CONV
    x = np.linspace(0.830, 1.103, 200); av = np.interp(x, Ts / tn, S5).mean(); return tn, av, av * tn, Ts.max() / tn
print('%-4s %-15s %6s %7s %8s %8s %6s' % ('tur', 'set', 'tohum', 'TN_C', 'dS5ort', 'carpim', 'Tmax'))
for kd in ('P', 'S', 'Z'):
    for n in sorted({r['set'] for r in R if r['kind'] == kd}):
        a = [r for r in R if r['kind'] == kd and r['set'] == n]
        for sd in sorted({r.get('seed', 0) for r in a}) + ['hep']:
            b = a if sd == 'hep' else [r for r in a if r.get('seed', 0) == sd]
            try: tn, av, pr, tm = ana(b); print('%-4s %-15s %6s %7.2f %8.4f %8.3f %6.3f' % (kd, n, sd, tn, av, pr, tm))
            except Exception as e: print('%-4s %-15s %6s eksik (%d) %s' % (kd, n, sd, len(b), str(e)[:30]))
for m in ('I', 'H'):
    for n in sorted({r['set'] for r in R if r['kind'] == 'B' and r['model'] == m}):
        a = [r for r in R if r['kind'] == 'B' and r['model'] == m and r['set'] == n]; Ls = sorted({r['L'] for r in a}); out = []
        for L1, L2 in zip(Ls[:-1], Ls[1:]):
            T = np.array(sorted({r['T'] for r in a if r['L'] == L1} & {r['T'] for r in a if r['L'] == L2}))
            u = lambda L: np.array([[r['U4'] for r in a if r['L'] == L and r['T'] == t][0] for t in T]); d = u(L1) - u(L2)
            k = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
            out.append('%d-%d: %s' % (L1, L2, ' '.join('%.2f' % (T[i] - d[i] * (T[i + 1] - T[i]) / (d[i + 1] - d[i])) for i in k) or 'yok'))
        print('B %s %-15s %s' % (m, n, ' | '.join(out)))
