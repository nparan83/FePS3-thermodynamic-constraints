"""run_mc5.py - FePS3 MC v5: iki yeni literatur J seti, MC v3 ile ayni protokol (S = 2 Blume-Capel, D = 2.66 meV, H || Ising ekseni, B = 0..6 T,
L = 24, 2 tohum, T = 0.5..1.5 T_N tahmini, 31 nokta, 8k + 30k tarama). fepslib2 (J2par/J2perp ayrimi; yama_fepslib2.py ile v1'den). g = 2 VARSAYIM.
Wildes20_r4_x2: Wildes JAP 127, 223903 (2020) Tablo I satir 4, yalniz bilineer kisim (biquadratic K Ising sinirinda ihmal), x2 (Lancon kurali).
Wildes20_r2_x2: ayni makale satir 2. Ilyas24_x2: Ilyas Nature 636, 609 (2024) Ext. Data Table 3, J -> -J, x2 VARSAYIM (dogrulanmadi).
T_N tahmini: yerel L16 on tarama (C tepesi). Cikti: mc5_raw.jsonl. ASCII-only."""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fepslib2 import mc_run
QUICK = os.environ.get('QUICK') == '1'
SETS = {'Wildes20_r4_x2': ((1.96, 1.96, -0.31, -0.90, 0.0), 92.0), 'Wildes20_r2_x2': ((1.80, 1.80, -0.094, -1.28, 0.0), 105.0),
        'Ilyas24_x2': ((1.12, 2.80, -0.02, -1.06, 0.0, -0.86), 100.0)}
D = 2.66; V5 = (-2, -1, 0, 1, 2); GMU = 0.115768; BS = (0, 1, 2, 3, 4, 5, 6)
NEQ, NME = (50, 150) if QUICK else (8000, 30000); NT = 3 if QUICK else 31; SEEDS = (1,) if QUICK else (1, 2); LL = 12 if QUICK else 24
if QUICK: BS = (0, 1, 6)

def task(t):
    name, T, seed, B = t; J, tn = SETS[name]
    r = mc_run(LL, J, V5, D, T, h=GMU * B, neq=NEQ, nmeas=NME, seed=7919 * seed + 13 * LL + int(10 * T) % 1009 + 101 * B, every=2, start='zz')
    r = {k: float(v) for k, v in r.items()}; r.update(set=name, T=float(T), seed=seed, B=B, TN=tn, L=LL); return r

if __name__ == '__main__':
    t0 = time.time(); nproc = int(os.environ.get('SLURM_NTASKS', '1'))
    done = set()
    if os.path.exists('mc5_raw.jsonl'):
        for l in open('mc5_raw.jsonl'):
            r = json.loads(l); done.add((r['set'], round(r['T'], 6), r['seed'], r['B']))
    tasks = [(n, float(T), s, B) for n, (J, tn) in SETS.items() for B in BS for s in SEEDS for T in np.linspace(0.5, 1.5, NT) * tn]
    tasks = [t for t in tasks if (t[0], round(t[1], 6), t[2], t[3]) not in done]
    print('kosu %d (onceden %d), cekirdek %d' % (len(tasks), len(done), nproc)); sys.stdout.flush()
    fh = open('mc5_raw.jsonl', 'a'); n = 0
    if nproc > 1:
        from multiprocessing import Pool
        pool = Pool(nproc); it = pool.imap_unordered(task, tasks, chunksize=1)
    else: it = map(task, tasks)
    for r in it:
        fh.write(json.dumps(r) + '\n'); fh.flush(); n += 1
        if n % 100 == 0: print('%d / %d  %.0f s' % (n, len(tasks), time.time() - t0)); sys.stdout.flush()
    if nproc > 1: pool.close(); pool.join()
    fh.close(); print('bitti %.0f s' % (time.time() - t0))
