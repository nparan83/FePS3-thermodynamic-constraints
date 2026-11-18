"""run_mc6.py - FePS3 MC v6, model sinirlari testi (fepslib3; D = 2.66 meV, g = 2 VARSAYIM). Cikti mc6_raw.jsonl (alan 'kind').
H: klasik Heisenberg + D (|S| = 2), 11 J seti, L 24, B 0/1/3/5/6 T, 25 T (0.70-1.30 T_N tahmini; tahmin yerel L16/24 on taramasi), 2 tohum, 4k + 16k.
3: Blume-Capel 3B, Lz 4, katmanlar arasi Jp = -0.0146 (Lancon x2) ve -0.146 meV (10x), Lancon + Wildes r4, B 0/1/3/5 T, 21 T (0.8-1.2 T_N Ising), 1 tohum.
Y: histerezis, Blume-Capel L 36, Lancon + Wildes r4, B 5 ve 6 T, isitma (zigzag baslangic) ve sogutma (rastgele baslangic), 25 adim 0.85-1.15 T_N.
ASCII-only."""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fepslib3 import heis_run, bcx_run
QUICK = os.environ.get('QUICK') == '1'
SETS = {'Wildes20_r4_x2': ((1.96, 1.96, -0.31, -0.90, 0.0), 59.2), 'Wildes20_r2_x2': ((1.80, 1.80, -0.094, -1.28, 0.0), 69.7),
        'Ilyas24_x2': ((1.12, 2.80, -0.02, -1.06, 0.0, -0.86), 72.0), 'Lancon_x2': ((2.92, 2.92, -0.08, -1.92, 0.0), 90.8),
        'Yang_U36': ((3.13, 3.13, -0.34, -2.01, 0.0), 100.8), 'Olsen_U2': ((2.10, 2.10, -0.21, -2.60, 0.0), 117.8),
        'Amir_U222': ((1.0025, -0.815, -0.31, -1.4275, 0.375), 125.3), 'own_U3.50': ((0.7479, 0.7479, -0.0026, -2.7487, 0.0), 108.2),
        'own_U3.00': ((0.9767, 0.9767, -0.0053, -3.1433, 0.0), 120.5), 'own_U2.25': ((1.5261, 1.5261, 0.0013, -3.9063, 0.0), 146.5),
        'Chat_U25': ((2.60, -2.20, -1.90, -0.60, 0.0), 154.0)}
TNI = {'Lancon_x2': 144.6, 'Wildes20_r4_x2': 88.4}
D = 2.66; GMU = 0.115768
LH, NTH, BH, SEEDS, EQ, ME = (12, 3, (0, 1, 5), (1,), 40, 120) if QUICK else (24, 25, (0, 1, 3, 5, 6), (1, 2), 4000, 16000)
L3, LZ, NT3, B3, JPS = (12, 2, 3, (0, 1, 5), (-0.0146,)) if QUICK else (24, 4, 21, (0, 1, 3, 5), (-0.0146, -0.146))
LY, NTY, BY, EQY, MEY = (12, 3, (5,), 30, 60) if QUICK else (36, 25, (5, 6), 2000, 4000)
if QUICK: SETS = {k: SETS[k] for k in ('Lancon_x2', 'Wildes20_r4_x2')}

def sd(*a):
    s = json.dumps(a); return int(sum((i + 1) * 7919 * ord(c) for i, c in enumerate(s)) % 2**31)

def task(t):
    k = t[0]
    if k == 'H':
        _, n, T, s, B = t; r = heis_run(LH, SETS[n][0], D, T, h=GMU * B, neq=EQ, nmeas=ME, seed=sd(t))
        r = {a: float(b) for a, b in r.items()}; r.update(kind='H', set=n, T=T, seed=s, B=B, TN=SETS[n][1]); return [r]
    if k == '3':
        _, n, jp, T, B = t; r = bcx_run(L3, SETS[n][0], D, T, h=GMU * B, Lz=LZ, Jp=jp, neq=EQ, nmeas=ME, seed=sd(t))
        r = {a: float(b) for a, b in r.items()}; r.update(kind='3', set=n, Jp=jp, T=T, B=B, TN=TNI[n]); return [r]
    _, n, B, dr = t; Ts = np.linspace(0.85, 1.15, NTY) * TNI[n]; out = []
    rs = np.random.RandomState(sd(t)); s0 = None if dr == 'up' else rs.randint(-2, 3, size=(1, LY, LY)).astype(float)
    for k2, T in enumerate(Ts if dr == 'up' else Ts[::-1]):
        r = bcx_run(LY, SETS[n][0], D, float(T), h=GMU * B, neq=EQY, nmeas=MEY, seed=sd(t, k2), s0=s0, ret=True); s0 = r.pop('state')
        r = {a: float(b) for a, b in r.items()}; r.update(kind='Y', set=n, B=B, dir=dr, step=k2, T=float(T), TN=TNI[n]); out.append(r)
    return out

def key(r):
    if r['kind'] == 'H': return ('H', r['set'], round(r['T'], 6), r['seed'], r['B'])
    if r['kind'] == '3': return ('3', r['set'], r['Jp'], round(r['T'], 6), r['B'])
    return ('Y', r['set'], r['B'], r['dir'])

if __name__ == '__main__':
    t0 = time.time(); nproc = int(os.environ.get('SLURM_NTASKS', '1')); done = {}
    if os.path.exists('mc6_raw.jsonl'):
        for l in open('mc6_raw.jsonl'):
            k = key(json.loads(l)); done[k] = done.get(k, 0) + 1
    tY = [('Y', n, B, d) for n in TNI for B in BY for d in ('up', 'down')]
    t3 = [('3', n, jp, float(T), B) for n in TNI for jp in JPS for B in B3 for T in np.linspace(0.8, 1.2, NT3) * TNI[n]]
    tH = [('H', n, float(T), s, B) for n, (J, tn) in SETS.items() for B in BH for s in SEEDS for T in np.linspace(0.7, 1.3, NTH) * tn]
    tasks = [t for t in tY if done.get(t, 0) < NTY] + [t for t in t3 if (t[0], t[1], t[2], round(t[3], 6), t[4]) not in done]
    tasks += [t for t in tH if (t[0], t[1], round(t[2], 6), t[3], t[4]) not in done]
    print('gorev %d (Y %d, 3B %d, H %d), cekirdek %d' % (len(tasks), len(tY), len(t3), len(tH), nproc)); sys.stdout.flush()
    fh = open('mc6_raw.jsonl', 'a'); n = 0
    if nproc > 1:
        from multiprocessing import Pool
        pool = Pool(nproc); it = pool.imap_unordered(task, tasks, chunksize=1)
    else: it = map(task, tasks)
    for rr in it:
        for r in rr: fh.write(json.dumps(r) + '\n')
        fh.flush(); n += 1
        if n % 100 == 0: print('%d / %d  %.0f s' % (n, len(tasks), time.time() - t0)); sys.stdout.flush()
    if nproc > 1: pool.close(); pool.join()
    fh.close(); print('bitti %.0f s' % (time.time() - t0))
