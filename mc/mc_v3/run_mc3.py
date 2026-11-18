"""run_mc3.py - FePS3 MC v3 'J setlerinin termodinamik sinavi': ayni S = 2 Blume-Capel (D = 2.66 meV), H || Ising ekseni, B = 0..6 T,
L = 24, 2 tohum, T = 0.5..1.5 T_N (31 nokta; T_N her set icin MC v1/v2 Binder sonucundan) -> M(T,B), C(T,B) -> Maxwell Delta S_M (yerelde).
own_U2.25 MC v2'de var (tekrarlanmaz). J kaynaklari: own = dft1/jmap_ozet.csv; literatur = FePS3_parametre_setleri.md (bag basina bir kez, S = 2).
Zeeman h = g muB B, g = 2 VARSAYIM. fepslib v1 (../mc_v1). Cikti: mc3_raw.jsonl (her satir bir kosu). ASCII-only."""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mc_v1'))
from fepslib import mc_run
QUICK = os.environ.get('QUICK') == '1'
SETS = {'own_U3.00': ((0.9767, 0.9767, -0.0053, -3.1433, 0.0), 185.69), 'own_U3.50': ((0.7479, 0.7479, -0.0026, -2.7487, 0.0), 163.40),
        'Lancon_x2': ((2.92, 2.92, -0.08, -1.92, 0.0), 144.6), 'Yang_U36': ((3.13, 3.13, -0.34, -2.01, 0.0), 161.1),
        'Olsen_U2': ((2.10, 2.10, -0.21, -2.60, 0.0), 178.8), 'Amir_U222': ((1.0025, -0.815, -0.31, -1.4275, 0.375), 184.0),
        'Chat_U25': ((2.60, -2.20, -1.90, -0.60, 0.0), 229.1)}
D = 2.66; V5 = (-2, -1, 0, 1, 2); GMU = 0.115768; BS = (0, 1, 2, 3, 4, 5, 6)
NEQ, NME = (100, 300) if QUICK else (8000, 30000); NT = 5 if QUICK else 31; SEEDS = (1,) if QUICK else (1, 2)
if QUICK: SETS = {k: SETS[k] for k in ('Lancon_x2', 'Amir_U222')}; BS = (0, 6)

def task(t):
    name, T, seed, B = t; J, tn = SETS[name]
    r = mc_run(24, J, V5, D, T, h=GMU * B, neq=NEQ, nmeas=NME, seed=7919 * seed + 13 * 24 + int(10 * T) % 1009 + 101 * B, every=2, start='zz')
    r = {k: float(v) for k, v in r.items()}; r.update(set=name, T=float(T), seed=seed, B=B, TN=tn); return r

if __name__ == '__main__':
    t0 = time.time(); nproc = int(os.environ.get('SLURM_NTASKS', '1'))
    done = set()
    if os.path.exists('mc3_raw.jsonl'):
        for l in open('mc3_raw.jsonl'):
            r = json.loads(l); done.add((r['set'], round(r['T'], 6), r['seed'], r['B']))
    tasks = [(n, float(T), s, B) for n, (J, tn) in SETS.items() for B in BS for s in SEEDS for T in np.linspace(0.5, 1.5, NT) * tn]
    tasks = [t for t in tasks if (t[0], round(t[1], 6), t[2], t[3]) not in done]
    print('kosu %d (onceden %d), cekirdek %d' % (len(tasks), len(done), nproc)); sys.stdout.flush()
    fh = open('mc3_raw.jsonl', 'a'); n = 0
    if nproc > 1:
        from multiprocessing import Pool
        pool = Pool(nproc); it = pool.imap_unordered(task, tasks, chunksize=1)
    else: it = map(task, tasks)
    for r in it:
        fh.write(json.dumps(r) + '\n'); fh.flush(); n += 1
        if n % 200 == 0: print('%d / %d  %.0f s' % (n, len(tasks), time.time() - t0)); sys.stdout.flush()
    if nproc > 1: pool.close(); pool.join()
    fh.close(); print('bitti %.0f s' % (time.time() - t0))
