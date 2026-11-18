# fepslib2.py v2 (J2par/J2perp) - FePS3 bal petegi J1par-J1perp-J2-J3-J4 Blume-Capel/Ising: geometri, EFT, MC. ASCII-only, yalniz numpy.
# H = - sum_bag J s_i s_j - D sum s_i^2 - h sum s_i  (her bag bir kez; J>0 FM; D>0 kolay eksen). meV, K.
# Orgu "tugla duvar": site (i,j), i zincir boyunca; up=(i+j)%2==0 dikey bagi j+1'e.
import numpy as np

KB = 0.08617333262
S3 = np.sqrt(3.0)
LABS = ('J1par', 'J1perp', 'J2', 'J3', 'J4', 'J2perp')

def coords(i, j):
    return i * S3 / 2.0, 1.5 * j + 0.5 * ((i + j) % 2 == 0)

def neighbor_offsets():
    out = {}
    for par in (0, 1):
        i0, j0 = (0, 0) if par == 0 else (1, 0)
        x0, y0 = coords(i0, j0); lst = []
        for di in range(-6, 7):
            for dj in range(-4, 5):
                if di == 0 and dj == 0: continue
                x, y = coords(i0 + di, j0 + dj); d = np.hypot(x - x0, y - y0)
                if abs(d - 1) < 1e-9: lab = 'J1par' if dj == 0 else 'J1perp'
                elif abs(d - S3) < 1e-9: lab = 'J2' if dj == 0 else 'J2perp'
                elif abs(d - 2) < 1e-9: lab = 'J3'
                elif abs(d - np.sqrt(7.0)) < 1e-9: lab = 'J4'
                else: continue
                lst.append((di, dj, lab))
        out[par] = lst
    return out

NB = neighbor_offsets()

def jvec(J):
    J = list(J) + [0.0] * (5 - len(J))
    J = J + [J[2]] if len(J) == 5 else J
    return dict(zip(LABS, J))

def active(J):
    Jd = jvec(J)
    return {p: [(di, dj, l) for di, dj, l in NB[p] if Jd[l] != 0.0] for p in (0, 1)}

def local_field(s, J, nb=None):
    nb = nb or active(J); Jd = jvec(J)
    i, j = np.indices(s.shape); par = (i + j) % 2; f = np.zeros_like(s, dtype=float)
    for p in (0, 1):
        fp = np.zeros_like(s, dtype=float)
        for di, dj, lab in nb[p]:
            fp += Jd[lab] * np.roll(np.roll(s, -di, 0), -dj, 1)
        f = np.where(par == p, fp, f)
    return f

def energy_per_site(s, J, D=0.0, h=0.0, nb=None):
    return (-0.5 * (s * local_field(s, J, nb)).sum() - D * (s**2).sum() - h * s.sum()) / s.size

def zigzag_patterns(Lx, Ly):
    pats = []
    for k in range(3):
        s = np.zeros((Lx, Ly), int); s[0, 0] = 1; st = [(0, 0)]; ok = True
        while st:
            i, j = st.pop(); up = (i + j) % 2 == 0
            for a, b, t in (((i + 1) % Lx, j, 1 if up else 2), ((i - 1) % Lx, j, 2 if up else 1),
                            (i, (j + (1 if up else -1)) % Ly, 0)):
                want = -s[i, j] if t == k else s[i, j]
                if s[a, b] == 0: s[a, b] = want; st.append((a, b))
                elif s[a, b] != want: ok = False
        pats.append(s if ok else None)
    return pats

def colors(Lx, Ly, J):
    """En az renkli (i%p, j%q) boyamasi; hicbir aktif ofset ayni renge dusmez."""
    nb = active(J)
    for p, q in ((3, 2), (4, 2), (6, 2), (3, 4), (4, 4), (6, 4), (12, 2), (12, 4)):
        if Lx % p or Ly % q: continue
        if all((di % p, dj % q) != (0, 0) for pp in (0, 1) for di, dj, l in nb[pp]):
            i, j = np.indices((Lx, Ly)); c = (i % p) * q + (j % q)
            return [c == k for k in range(p * q)]
    raise ValueError('uygun boyama yok: L=%d' % Lx)

def site_moments(vals, x, beta, D=0.0):
    v = np.asarray(vals, float); n = len(v) - 1
    E = beta * (np.outer(np.atleast_1d(x), v) + D * v**2)
    E -= E.max(axis=1, keepdims=True); w = np.exp(E); Z = w.sum(axis=1)
    return np.array([(w * v**k).sum(axis=1) / Z for k in range(1, n + 1)])

def probs_from_moments(vals, mom):
    v = np.asarray(vals, float); n = len(v)
    V = np.vstack([v**k for k in range(n)])
    p = np.clip(np.linalg.solve(V, np.r_[1.0, mom]), 0, None)
    return p / p.sum()

class HoneyEFT:
    """Kaneyoshi ayristirmasi (komsular bagimsiz, dagilim momentlerden); kabuk gruplari tam kivrimla."""
    def __init__(self, J, vals=(-2, -1, 0, 1, 2), D=0.0, px=2, py=2):
        self.v = np.array(vals, float); self.D = D; self.nm = len(self.v) - 1; self.Jd = jvec(J)
        pat = lambda i, j: (i % px) * py + (j % py)
        self.sub = sorted({pat(i, j) for i in range(px) for j in range(py)}); self.groups = {}
        nb = active(J)
        for i in range(px):
            for j in range(py):
                a = pat(i, j); g = {}
                for di, dj, lab in nb[(i + j) % 2]:
                    k = (lab, pat((i + di) % px, (j + dj) % py)); g[k] = g.get(k, 0) + 1
                self.groups[a] = g

    def _field_dist(self, a, P):
        xs, ps = np.array([0.0]), np.array([1.0])
        for (lab, b), n in self.groups[a].items():
            vals, pr = np.array([0.0]), np.array([1.0])
            for _ in range(n):
                vals = (vals[:, None] + self.v[None, :]).ravel(); pr = (pr[:, None] * P[b][None, :]).ravel()
                u, inv = np.unique(np.round(vals, 9), return_inverse=True); pr = np.bincount(inv, pr); vals = u
            xn = (xs[:, None] + self.Jd[lab] * vals[None, :]).ravel(); pn = (ps[:, None] * pr[None, :]).ravel()
            u, inv = np.unique(np.round(xn, 9), return_inverse=True); ps = np.bincount(inv, pn); xs = u
        return xs, ps

    def step(self, M, beta, h=0.0):
        P = {b: probs_from_moments(self.v, M[b]) for b in self.sub}; out = {}
        for a in self.sub:
            xs, ps = self._field_dist(a, P); out[a] = site_moments(self.v, xs + h, beta, self.D) @ ps
        return out

def eft_lambda(m, T):
    beta = 1.0 / (KB * T); odd = list(range(0, m.nm, 2))
    M = {a: site_moments(m.v, np.array([0.0]), beta, m.D)[:, 0] for a in m.sub}
    for a in m.sub: M[a][odd] = 0.0
    for _ in range(500):
        N = m.step(M, beta)
        for a in m.sub: N[a][odd] = 0.0
        d = max(abs(N[a] - M[a]).max() for a in m.sub); M = N
        if d < 1e-13: break
    keys = [(a, k) for a in m.sub for k in odd]; n = len(keys); Jm = np.zeros((n, n)); e = 1e-7
    base = m.step(M, beta)
    for c, (a, k) in enumerate(keys):
        Mp = {b: M[b].copy() for b in m.sub}; Mp[a][k] += e; Np = m.step(Mp, beta)
        Jm[:, c] = [(Np[b][kk] - base[b][kk]) / e for (b, kk) in keys]
    return max(abs(np.linalg.eigvals(Jm)))

def eft_Tc(m, lo=1.0, hi=3000.0, it=45):
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if eft_lambda(m, mid) > 1: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)

def mc_run(L, J, vals, D, T, h=0.0, neq=2000, nmeas=8000, seed=1, every=2, start='zz', keepE=0):
    """Renkli Metropolis. Doner: dict(e, m, mzz, U4 (zigzag), U4m (uniform), C, chi, chizz[, Eser])."""
    rs = np.random.RandomState(seed); v = np.asarray(vals, float); nb = active(J); Jd = jvec(J)
    cols = colors(L, L, J); pats = [p for p in zigzag_patterns(L, L) if p is not None]
    assert len(pats) == 3, 'L=%d uc zigzag yonelimini tasimiyor (L 12nin kati olmali)' % L
    s = v.max() * (pats[0].astype(float) if start == 'zz' else np.ones((L, L)))
    N = s.size; beta = 1.0 / (KB * T); i, j = np.indices((L, L)); par = (i + j) % 2
    El, Ml, Zl = [], [], []
    for sw in range(neq + nmeas):
        for cm in cols:
            f = np.zeros_like(s)
            for p in (0, 1):
                fp = np.zeros_like(s)
                for di, dj, lab in nb[p]:
                    fp += Jd[lab] * np.roll(np.roll(s, -di, 0), -dj, 1)
                f = np.where(par == p, fp, f)
            n = cm.sum(); old = s[cm]; new = v[rs.randint(len(v), size=n)]
            dE = -(new - old) * (f[cm] + h) - D * (new**2 - old**2)
            acc = rs.random_sample(n) < np.exp(-beta * np.clip(dE, 0, None))
            s[cm] = np.where(acc, new, old)
        if sw >= neq and (sw - neq) % every == 0:
            El.append(energy_per_site(s, J, D, h, nb)); Ml.append(s.mean())
            Zl.append(max(abs((s * p).mean()) for p in pats))
    E, Mu, Zz = np.array(El), np.array(Ml), np.array(Zl)
    m2, m4 = (Zz**2).mean(), (Zz**4).mean(); u2, u4 = (Mu**2).mean(), (Mu**4).mean()
    r = dict(e=E.mean(), m=Mu.mean(), mzz=Zz.mean(), U4=1 - m4 / (3 * m2 * m2),
             U4m=(1 - u4 / (3 * u2 * u2)) if u2 > 0 else 0.0, C=N * E.var() / (KB * T**2),
             chi=N * Mu.var() / (KB * T), chizz=N * Zz.var() / (KB * T))
    if keepE: r['Eser'] = E[::keepE].astype(np.float32)
    return r

def binder_cross(Ts, Ua, Ub, umin=0.30, umax=0.65):
    """Iki L arasi U4 kesisimi, komsu noktalar arasi dogrusal interpolasyon (LNMO v1.1/v1.2).
    Yalniz kesisim degeri umin..umax araligindaysa kabul (dusuk-T platosu gurultusu). Doner (Tc, U*, bayrak)."""
    Ts = np.asarray(Ts, float); d = np.asarray(Ua) - np.asarray(Ub); out = []
    for k in np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]:
        t = d[k] / (d[k] - d[k + 1]); tc = Ts[k] + (Ts[k + 1] - Ts[k]) * t
        us = Ua[k] + (Ua[k + 1] - Ua[k]) * t
        if umin <= us <= umax: out.append((tc, us))
    if not out: return (np.nan, np.nan, 'yok')
    return (float(np.mean([o[0] for o in out])), float(np.mean([o[1] for o in out])), 'ok' if len(out) == 1 else 'coklu')
