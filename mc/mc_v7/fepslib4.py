"""fepslib4.py - FePS3 MC v7: heisk_run = fepslib3.heis_run + biquadratic K (S_i.S_j)^2 on chosen bond labels + block errors.
H = -sum_bag J Si.Sj + sum_bag K (Si.Sj)^2 - D sum (Siz)^2 - h sum Siz (bag = bond, each once). K = None -> heis_run ile ayni RNG dizisi.
m_se, e_se: 10 blokluk standart hata. ASCII-only, yalniz numpy."""
import numpy as np
from fepslib2 import active, jvec, colors, zigzag_patterns, KB
from fepslib3 import nbfield, _obs, bcx_run

def _blk(x, nb=10):
    x = np.asarray(x); n = len(x) // nb
    if n < 2: return 0.0
    b = x[:n * nb].reshape(nb, n).mean(1); return float(b.std(ddof=1) / np.sqrt(nb))

def _kdot(s, nbk, par):
    """her site icin K-baglari uzerinden sum_K K (s_i.s_j)^2 (site basina, cift sayim)."""
    out = np.zeros(s.shape[1:])
    for p in (0, 1):
        acc = np.zeros(s.shape[1:])
        for di, dj, lab, K in nbk[p]:
            sj = np.roll(np.roll(s, -di, -2), -dj, -1); acc += K * (s * sj).sum(0)**2
        out = np.where(par == p, acc, out)
    return out

def _kdiff(s, new_full, nbk, par, cm):
    d = np.zeros(s.shape[1:])
    for p in (0, 1):
        acc = np.zeros(s.shape[1:])
        for di, dj, lab, K in nbk[p]:
            sj = np.roll(np.roll(s, -di, -2), -dj, -1)
            acc += K * ((new_full * sj).sum(0)**2 - (s * sj).sum(0)**2)
        d = np.where(par == p, acc, d)
    return d[cm]

def heisk_run(L, J, D, T, h=0.0, K=None, Smag=2.0, neq=2000, nmeas=8000, seed=1, every=2, start='zz', dlt=0.6):
    rs = np.random.RandomState(seed); nb = active(J); Jd = jvec(J)
    i, j = np.indices((L, L)); par = (i + j) % 2; pats = [p for p in zigzag_patterns(L, L) if p is not None]
    assert len(pats) == 3
    K = K or {}; nbk = {p: [(di, dj, l, K[l]) for di, dj, l in nb[p] if K.get(l, 0.0) != 0.0] for p in (0, 1)}; useK = any(nbk[p] for p in (0, 1))
    s = np.zeros((3, L, L)); s[2] = Smag * (pats[0] if start == 'zz' else np.ones((L, L)))
    cols = colors(L, L, J); N = L * L; beta = 1.0 / (KB * T); El, Ml, Zl = [], [], []; acc_n = acc_t = 0
    for sw in range(neq + nmeas):
        for cm in cols:
            for mv in (0, 1):
                f = nbfield(s, nb, Jd, par); f[2] += h; old = s[:, cm]; n = old.shape[1]
                if mv == 0:
                    new = old + dlt * Smag * rs.standard_normal((3, n)); new *= Smag / np.linalg.norm(new, axis=0)
                else:
                    new = old.copy(); new[2] = -new[2]
                dE = -((new - old) * f[:, cm]).sum(0) - D * (new[2]**2 - old[2]**2)
                if useK:
                    nf = s.copy(); nf[:, cm] = new; dE = dE + _kdiff(s, nf, nbk, par, cm)
                acc = rs.random_sample(n) < np.exp(-beta * np.clip(dE, 0, None)); s[:, cm] = np.where(acc, new, old)
                if mv == 0: acc_n += acc.sum(); acc_t += n
        if sw < neq and sw % 50 == 49:
            a = acc_n / max(acc_t, 1); dlt = min(2.0, max(0.02, dlt * (1.15 if a > 0.5 else 0.87))); acc_n = acc_t = 0
        if sw >= neq and (sw - neq) % every == 0:
            f = nbfield(s, nb, Jd, par); eK = 0.5 * _kdot(s, nbk, par).sum() if useK else 0.0
            El.append((-0.5 * (s * f).sum() + eK - D * (s[2]**2).sum() - h * s[2].sum()) / N); Ml.append(s[2].mean())
            Zl.append(max(abs((s[2] * p).mean()) for p in pats))
    r = _obs(El, Ml, Zl, N, T); r['dlt'] = dlt; r['m_se'] = _blk(Ml); r['e_se'] = _blk(El)
    return r
