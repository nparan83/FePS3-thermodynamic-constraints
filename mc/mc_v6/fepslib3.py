"""fepslib3.py - FePS3 MC v6 motorlari (fepslib2 geometrisi uzerine). ASCII-only, yalniz numpy.
heis_run: klasik Heisenberg + D (vektor spin |S| = Smag), H = -sum_bag J Si.Sj - D sum (Siz)^2 - h sum Siz.
bcx_run : Blume-Capel S = 2, Lz katman (katmanlar arasi Jp, ayni (i,j) ustu/alti), J1 baglarinda biquadratic K1,
          J2 baglarinda K2: -K (Siz Sjz)^2 ; baslangic durumu s0 verilebilir, son durum dondurulebilir (histerezis)."""
import numpy as np
from fepslib2 import active, jvec, colors, zigzag_patterns, KB

def nbfield(s, nb, w, par):
    """s: (..., L, L); w: etiket -> agirlik. Duzlem ici komsu alani."""
    f = np.zeros_like(s, dtype=float)
    for p in (0, 1):
        fp = np.zeros_like(s, dtype=float)
        for di, dj, lab in nb[p]:
            if w.get(lab, 0.0) != 0.0: fp += w[lab] * np.roll(np.roll(s, -di, -2), -dj, -1)
        f = np.where(par == p, fp, f)
    return f

def _obs(El, Ml, Zl, N, T):
    E, Mu, Zz = np.array(El), np.array(Ml), np.array(Zl)
    m2, m4 = (Zz**2).mean(), (Zz**4).mean(); u2, u4 = (Mu**2).mean(), (Mu**4).mean()
    return dict(e=E.mean(), m=Mu.mean(), mzz=Zz.mean(), U4=1 - m4 / (3 * m2 * m2), U4m=(1 - u4 / (3 * u2 * u2)) if u2 > 0 else 0.0,
                C=N * E.var() / (KB * T**2), chi=N * Mu.var() / (KB * T), chizz=N * Zz.var() / (KB * T))

def bcx_run(L, J, D, T, h=0.0, Lz=1, Jp=0.0, K1=0.0, K2=0.0, neq=2000, nmeas=8000, seed=1, every=2, s0=None, ret=False):
    rs = np.random.RandomState(seed); v = np.array([-2., -1., 0., 1., 2.]); nb = active(J); Jd = jvec(J)
    i, j = np.indices((L, L)); par = (i + j) % 2; pats = [p for p in zigzag_patterns(L, L) if p is not None]
    assert len(pats) == 3 and (Lz == 1 or Lz % 2 == 0)
    zs = (np.where(np.arange(Lz) % 2 == 0, 1.0, -1.0) if Jp < 0 else np.ones(Lz))[:, None, None]
    s = (2.0 * pats[0][None, :, :] * zs).astype(float) if s0 is None else np.array(s0, float).reshape(Lz, L, L)
    wK = {'J1par': K1, 'J1perp': K1, 'J2': K2, 'J2perp': K2}; useK = (K1 != 0.0) or (K2 != 0.0)
    cin = colors(L, L, J); z = np.arange(Lz)[:, None, None]
    cols = [c[None, :, :] & (z % 2 == k) for c in cin for k in ((0, 1) if Lz > 1 else (0,))]
    def fld(s):
        f = nbfield(s, nb, Jd, par)
        if Lz > 1: f = f + Jp * (np.roll(s, 1, 0) + np.roll(s, -1, 0))
        return f
    N = s.size; beta = 1.0 / (KB * T); El, Ml, Zl = [], [], []
    for sw in range(neq + nmeas):
        for cm in cols:
            f = fld(s); q = nbfield(s * s, nb, wK, par) if useK else 0.0
            n = cm.sum(); old = s[cm]; new = v[rs.randint(5, size=n)]
            dE = -(new - old) * (f[cm] + h) - (D + (q[cm] if useK else 0.0)) * (new**2 - old**2)
            acc = rs.random_sample(n) < np.exp(-beta * np.clip(dE, 0, None)); s[cm] = np.where(acc, new, old)
        if sw >= neq and (sw - neq) % every == 0:
            f = fld(s); q = nbfield(s * s, nb, wK, par) if useK else 0.0
            El.append((-0.5 * (s * f).sum() - 0.5 * (s * s * q).sum() - D * (s * s).sum() - h * s.sum()) / N); Ml.append(s.mean())
            Zl.append(max(abs((s * p[None] * zs).mean()) for p in pats) if Jp != 0.0 or Lz == 1 else
                      max(np.mean([abs((s[k] * p).mean()) for k in range(Lz)]) for p in pats))
    r = _obs(El, Ml, Zl, N, T)
    if ret: r['state'] = s.astype(np.int8)
    return r

def heis_run(L, J, D, T, h=0.0, Smag=2.0, neq=2000, nmeas=8000, seed=1, every=2, start='zz', dlt=0.6):
    rs = np.random.RandomState(seed); nb = active(J); Jd = jvec(J)
    i, j = np.indices((L, L)); par = (i + j) % 2; pats = [p for p in zigzag_patterns(L, L) if p is not None]
    assert len(pats) == 3
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
                acc = rs.random_sample(n) < np.exp(-beta * np.clip(dE, 0, None)); s[:, cm] = np.where(acc, new, old)
                if mv == 0: acc_n += acc.sum(); acc_t += n
        if sw < neq and sw % 50 == 49:
            a = acc_n / max(acc_t, 1); dlt = min(2.0, max(0.02, dlt * (1.15 if a > 0.5 else 0.87))); acc_n = acc_t = 0
        if sw >= neq and (sw - neq) % every == 0:
            f = nbfield(s, nb, Jd, par)
            El.append((-0.5 * (s * f).sum() - D * (s[2]**2).sum() - h * s[2].sum()) / N); Ml.append(s[2].mean())
            Zl.append(max(abs((s[2] * p).mean()) for p in pats))
    r = _obs(El, Ml, Zl, N, T); r['dlt'] = dlt
    return r
