"""run_mc7.py - FePS3 MC v7 (makale 1). Cikti mc7_raw.jsonl (alan 'kind'). D = 2.66 meV disinda yayimlanmis D kullanilan setler belirtildi. ASCII-only.
P: yayimlanmis Hamiltonyen, Heisenberg+D(+K), L 24, B 0/1/3/5 T, 25 T (0.70-1.30 x on tarama T_N), 2 tohum, 4k + 16k.
B: Binder, B = 0, L 24/36/48, Ising (Blume-Capel) ve Heisenberg (D 2.66), 11 set, 15 T (0.92-1.08 x T_N), 1 tohum; L48 6k + 16k.
Z: Wildes fit 4, 3B Lz 4, J' = -0.146, B 0/1/3/5 T, 25 T (0.75-1.30 x 88.4 K), 4k + 16k.
S: ek tohum (3, 4), Heisenberg D 2.66, 4 set, B 0/1/3/5 T, MC v6 izgarasi (0.70-1.30 x T_N v6 tahmini)."""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fepslib3 import bcx_run
from fepslib4 import heisk_run
QUICK = os.environ.get('QUICK') == '1'
SETS = {'Wildes20_r4_x2': (1.96, 1.96, -0.31, -0.90, 0.0), 'Wildes20_r2_x2': (1.80, 1.80, -0.094, -1.28, 0.0),
        'Ilyas24_x2': (1.12, 2.80, -0.02, -1.06, 0.0, -0.86), 'Lancon_x2': (2.92, 2.92, -0.08, -1.92, 0.0),
        'Yang_U36': (3.13, 3.13, -0.34, -2.01, 0.0), 'Olsen_U2': (2.10, 2.10, -0.21, -2.60, 0.0),
        'Amir_U222': (1.0025, -0.815, -0.31, -1.4275, 0.375), 'own_U3.50': (0.7479, 0.7479, -0.0026, -2.7487, 0.0),
        'own_U3.00': (0.9767, 0.9767, -0.0053, -3.1433, 0.0), 'own_U2.25': (1.5261, 1.5261, 0.0013, -3.9063, 0.0),
        'Chat_U25': (2.60, -2.20, -1.90, -0.60, 0.0)}
TNI = {'Wildes20_r4_x2': 88.4, 'Wildes20_r2_x2': 101.8, 'Ilyas24_x2': 103.5, 'Lancon_x2': 144.6,
        'Yang_U36': 161.1, 'Olsen_U2': 178.8, 'Amir_U222': 184.0, 'own_U3.50': 163.4,
        'own_U3.00': 185.7, 'own_U2.25': 228.9, 'Chat_U25': 229.1}
TNH = {'Wildes20_r4_x2': 59.5, 'Wildes20_r2_x2': 68.3, 'Ilyas24_x2': 68.7, 'Lancon_x2': 90.3,
        'Yang_U36': 100.1, 'Olsen_U2': 114.9, 'Amir_U222': 121.7, 'own_U3.50': 106.3,
        'own_U3.00': 119.4, 'own_U2.25': 143.1, 'Chat_U25': 145.7}
TNE6 = {'Wildes20_r4_x2': 59.2, 'Lancon_x2': 90.8, 'own_U3.00': 120.5, 'Olsen_U2': 117.8}
PUB = {'W4pub': ('Wildes20_r4_x2', 2.52, {'J1par': 0.115, 'J1perp': 0.115, 'J2': 0.02125, 'J2perp': 0.02125}, 55.0),
       'W2pub': ('Wildes20_r2_x2', 2.53, {'J1par': 0.1375, 'J1perp': 0.1375}, 60.0), 'OlsenD': ('Olsen_U2', 0.101, {}, 85.0),
       'YangD': ('Yang_U36', 4.85, {}, 120.0), 'AmirDB': ('Amir_U222', 0.2225, {'J1par': -0.13125, 'J1perp': -0.13125}, 95.0)}
GMU = 0.115768
NP_, BP, SP, EQ, ME = (3, (0, 1, 5), (1,), 30, 60) if QUICK else (25, (0, 1, 3, 5), (1, 2), 4000, 16000)
LB, NB_, EQB = ((8, 12), 3, (30, 30)) if QUICK else ((24, 36, 48), 15, (4000, 4000, 6000))
NZ, BZ = (3, (0, 5)) if QUICK else (25, (0, 1, 3, 5))
LL = 12 if QUICK else 24

def sd(*a):
    s = json.dumps(a); return int(sum((i + 1) * 7919 * ord(c) for i, c in enumerate(s)) % 2**31)

def flt(r): return {a: float(b) for a, b in r.items()}

def task(t):
    k = t[0]
    if k == 'P':
        _, n, T, s, B = t; base, D, K, tn = PUB[n]
        r = flt(heisk_run(LL, SETS[base], D, T, h=GMU * B, K=K, neq=EQ, nmeas=ME, seed=sd(t))); r.update(kind='P', set=n, T=T, seed=s, B=B, TN=tn, D=D)
    elif k == 'B':
        _, mdl, n, L, T = t; il = LB.index(L)
        if mdl == 'I': r = flt(bcx_run(L, SETS[n], 2.66, T, neq=EQB[il], nmeas=ME, seed=sd(t)))
        else: r = flt(heisk_run(L, SETS[n], 2.66, T, neq=EQB[il], nmeas=ME, seed=sd(t)))
        r.update(kind='B', model=mdl, set=n, L=L, T=T)
    elif k == 'Z':
        _, T, B = t; r = flt(bcx_run(LL, SETS['Wildes20_r4_x2'], 2.66, T, h=GMU * B, Lz=2 if QUICK else 4, Jp=-0.146, neq=EQ, nmeas=ME, seed=sd(t)))
        r.update(kind='Z', set='Wildes20_r4_x2', Jp=-0.146, T=T, B=B)
    else:
        _, n, T, s, B = t; r = flt(heisk_run(LL, SETS[n], 2.66, T, h=GMU * B, neq=EQ, nmeas=ME, seed=sd(t))); r.update(kind='S', set=n, T=T, seed=s, B=B)
    return r

def key(r):
    if r['kind'] in ('P', 'S'): return (r['kind'], r['set'], round(r['T'], 6), r['seed'], r['B'])
    if r['kind'] == 'B': return ('B', r['model'], r['set'], r['L'], round(r['T'], 6))
    return ('Z', round(r['T'], 6), r['B'])

if __name__ == '__main__':
    t0 = time.time(); nproc = int(os.environ.get('SLURM_NTASKS', '1')); done = set()
    if os.path.exists('mc7_raw.jsonl'):
        for l in open('mc7_raw.jsonl'): done.add(key(json.loads(l)))
    sets = list(SETS) if not QUICK else ['Lancon_x2', 'Wildes20_r4_x2']
    tB = [('B', m, n, L, float(T)) for L in LB[::-1] for m in ('I', 'H') for n in sets for T in np.linspace(0.92, 1.08, NB_) * (TNI[n] if m == 'I' else TNH[n])]
    tP = [('P', n, float(T), s, B) for n in (PUB if not QUICK else ['W4pub']) for B in BP for s in SP for T in np.linspace(0.70, 1.30, NP_) * PUB[n][3]]
    tZ = [('Z', float(T), B) for B in BZ for T in np.linspace(0.75, 1.30, NZ) * 88.4]
    tS = [('S', n, float(T), s, B) for n in (TNE6 if not QUICK else ['Lancon_x2']) for B in BP for s in (3, 4) for T in np.linspace(0.70, 1.30, NP_) * TNE6[n]]
    allt = tB + tP + tZ + tS
    tasks = [t for t in allt if (t[:4] + (round(t[4], 6),) if t[0] == 'B' else ((t[0], round(t[1], 6), t[2]) if t[0] == 'Z' else (t[0], t[1], round(t[2], 6), t[3], t[4]))) not in done]
    print('gorev %d / %d (B %d, P %d, Z %d, S %d), cekirdek %d' % (len(tasks), len(allt), len(tB), len(tP), len(tZ), len(tS), nproc)); sys.stdout.flush()
    fh = open('mc7_raw.jsonl', 'a'); n = 0
    if nproc > 1:
        from multiprocessing import Pool
        pool = Pool(nproc); it = pool.imap_unordered(task, tasks, chunksize=1)
    else: it = map(task, tasks)
    for r in it:
        fh.write(json.dumps(r) + '\n'); fh.flush(); n += 1
        if n % 100 == 0: print('%d / %d  %.0f s' % (n, len(tasks), time.time() - t0)); sys.stdout.flush()
    if nproc > 1: pool.close(); pool.join()
    fh.close(); print('bitti %.0f s' % (time.time() - t0))
