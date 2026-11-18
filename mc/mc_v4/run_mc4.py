"""run_mc4.py - FePS3 MC v4 sonlu boyut testi: Maxwell Delta S_M(6 T) ve dchi/dT L'ye bagli mi? fepslib v1 (../mc_v1), D = 2.66 meV, g = 2 VARSAYIM.
Setler: Lancon_x2 (L 16, 36; L 24 MC v3'te ayni T izgarasinda var) ve own_U2.25 (L 16, 24, 36; MC v2 izgarasi farkli oldugu icin L 24 yeniden).
T = 0.7..1.3 T_N (19 nokta, adim T_N/30 = MC v3 izgarasi), B = 0..6 T, 2 tohum, 8k + 30k tarama. Cikti: mc4_raw.jsonl. ASCII-only."""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mc_v1'))
from fepslib import mc_run
QUICK = os.environ.get('QUICK') == '1'
SETS = {'Lancon_x2': ((2.92, 2.92, -0.08, -1.92, 0.0), 144.6, (36, 16)), 'own_U2.25': ((1.5261, 1.5261, 0.0013, -3.9063, 0.0), 228.9, (36, 24, 16))}
D = 2.66; V5 = (-2, -1, 0, 1, 2); GMU = 0.115768; BS = (0, 1, 2, 3, 4, 5, 6)
NEQ, NME = (50, 150) if QUICK else (8000, 30000); NT = 3 if QUICK else 19; SEEDS = (1,) if QUICK else (1, 2)
if QUICK: BS = (0, 6); SETS = {k: (v[0], v[1], (8,)) for k, v in SETS.items()}

def task(t):
    name, L, T, seed, B = t; J, tn, Ls = SETS[name]
    r = mc_run(L, J, V5, D, T, h=GMU * B, neq=NEQ, nmeas=NME, seed=7919 * seed + 13 * L + int(10 * T) % 1009 + 101 * B, every=2, start='zz')
    r = {k: float(v) for k, v in r.items()}; r.update(set=name, L=L, T=float(T), seed=seed, B=B, TN=tn); return r

if __name__ == '__main__':
    t0 = time.time(); nproc = int(os.environ.get('SLURM_NTASKS', '1'))
    done = set()
    if os.path.exists('mc4_raw.jsonl'):
        for l in open('mc4_raw.jsonl'):
            r = json.loads(l); done.add((r['set'], r['L'], round(r['T'], 6), r['seed'], r['B']))
    tasks = [(n, L, float(T), s, B) for n, (J, tn, Ls) in SETS.items() for L in Ls for B in BS for s in SEEDS for T in np.linspace(0.7, 1.3, NT) * tn]
    tasks = [t for t in tasks if (t[0], t[1], round(t[2], 6), t[3], t[4]) not in done]
    tasks.sort(key=lambda t: -t[1])
    print('kosu %d (onceden %d), cekirdek %d' % (len(tasks), len(done), nproc)); sys.stdout.flush()
    fh = open('mc4_raw.jsonl', 'a'); n = 0
    if nproc > 1:
        from multiprocessing import Pool
        pool = Pool(nproc); it = pool.imap_unordered(task, tasks, chunksize=1)
    else: it = map(task, tasks)
    for r in it:
        fh.write(json.dumps(r) + '\n'); fh.flush(); n += 1
        if n % 100 == 0: print('%d / %d  %.0f s' % (n, len(tasks), time.time() - t0)); sys.stdout.flush()
    if nproc > 1: pool.close(); pool.join()
    fh.close(); print('bitti %.0f s' % (time.time() - t0))
