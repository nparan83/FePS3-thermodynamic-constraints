"""ozet6.py - mc6_raw.jsonl hizli ozet (login node). H/3: T_N (B = 0 C tepesi, parabol), 5 T Maxwell Delta S_M tepe ve 0.830-1.103 T_N
ortalamasi (J/kgK), 1 T dchi/dT en buyuk. Y: isitma-sogutma m farkinin en buyugu (mu_B/Fe) ve yeri. g = 2, M_mol 183.0. ASCII-only."""
import json
import numpy as np
GMU = 0.115768; CONV = 8.314462618 / 0.08617333262 / 0.183; TRAP = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
R = [json.loads(l) for l in open('mc6_raw.jsonl')]
def ana(a):
    Ts = np.array(sorted({round(r['T'], 4) for r in a})); Bs = sorted({r['B'] for r in a})
    g = lambda b, k: np.array([np.mean([r[k] for r in a if r['B'] == b and abs(r['T'] - T) < 1e-3] or [np.nan]) for T in Ts])
    M = np.array([g(b, 'm') for b in Bs]); C0 = g(0, 'C'); j = int(np.nanargmax(C0)); jj = slice(max(j - 2, 0), j + 3)
    pc = np.polyfit(Ts[jj], C0[jj], 2); tn = -pc[1] / (2 * pc[0]); i5 = Bs.index(5)
    S5 = TRAP(np.gradient(M[:i5 + 1], Ts, axis=1), GMU * np.array(Bs[:i5 + 1]), axis=0) * CONV
    x = np.linspace(0.830, 1.103, 200); av = np.interp(x, Ts / tn, S5).mean()
    return tn, S5.max(), av, np.gradient(2 * M[Bs.index(1)], Ts).max(), len(a)
print('%-4s %-15s %8s %6s %7s %8s %8s %10s' % ('tur', 'set', 'Jp', 'kosu', 'TN_C', 'dS5max', 'dS5ort', 'dchidT'))
for kd in ('H', '3'):
    for n, jp in sorted({(r['set'], r.get('Jp', 0.0)) for r in R if r['kind'] == kd}):
        a = [r for r in R if r['kind'] == kd and r['set'] == n and r.get('Jp', 0.0) == jp]
        try: tn, pk, av, dc, nn = ana(a); print('%-4s %-15s %8.4f %6d %7.1f %8.4f %8.4f %10.3e' % (kd, n, jp, nn, tn, pk, av, dc))
        except Exception as e: print('%-4s %-15s eksik (%d) %s' % (kd, n, len(a), str(e)[:40]))
for n, B in sorted({(r['set'], r['B']) for r in R if r['kind'] == 'Y'}):
    up = {round(r['T'], 3): r for r in R if r['kind'] == 'Y' and r['set'] == n and r['B'] == B and r['dir'] == 'up'}
    dn = {round(r['T'], 3): r for r in R if r['kind'] == 'Y' and r['set'] == n and r['B'] == B and r['dir'] == 'down'}
    Tc = sorted(set(up) & set(dn)); d = [2 * abs(up[T]['m'] - dn[T]['m']) for T in Tc]; z = [abs(up[T]['mzz'] - dn[T]['mzz']) for T in Tc]
    if Tc: print('Y    %-15s B=%d T  adim %d  max|dM| %.4f muB/Fe @ %.1f K  max|dmzz| %.3f' % (n, B, len(Tc), max(d), Tc[int(np.argmax(d))], max(z)))
