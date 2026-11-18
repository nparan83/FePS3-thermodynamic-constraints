import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'analysis')); sys.path.insert(0, os.path.join(REPO, 'mc', 'lib'))
def P(f):
    for d in ('data/experiment', 'data/derived', 'mc/lib', 'analysis'):
        p = os.path.join(REPO, d, f)
        if os.path.exists(p): return p
    raise FileNotFoundError(f)

import os, glob, json, tarfile, hashlib
import numpy as np
import pandas as pd

# Global state
pd.set_option('display.max_colwidth', 70)

# Load ana7.py helpers
exec(open(P('ana7.py'), encoding='utf-8').read())

# Load dependency CSVs
mf = pd.read_csv(P('MF_kisitlar.csv'))
spinflip_alanlari = pd.read_csv(P('spinflip_alanlari.csv'))
LSWT_kontrol = pd.read_csv(P('LSWT_kontrol.csv'))
Q = pd.read_csv(P('MCE_5T_deney_karsilastirma.csv'))

# Constants
GMU = 0.115768
CONV = 8.314462618 / 0.08617333262 / 0.183
WIN = {'ana': (0.830, 1.103), 'dar': (0.90, 1.05), 'genis': (0.75, 1.20)}
dEexp = 9.3367

# Load mc_v7 tar
od0 = os.path.join(REPO, 'mc')
p = os.path.join(od0, 'mc_v7', 'mc_v7_sonuc.tar.gz')
os.makedirs('v7', exist_ok=True)
with tarfile.open(p) as t:
    t.extractall('v7')
R7 = pd.DataFrame([json.loads(l) for l in open('v7/mc7_raw.jsonl')])

# Load mc_v6
p6 = os.path.join(od0, 'mc_v6', 'mc_v6_sonuc.tar.gz')
os.makedirs('v6', exist_ok=True)
with tarfile.open(p6) as t:
    names = [m.name for m in t.getmembers()]
    t.extractall('v6')
f6 = [n for n in names if n.endswith('.jsonl')]
R6 = pd.concat([pd.DataFrame([json.loads(l) for l in open('v6/' + n)]).assign(src=n) for n in f6])

# Load Ising runs from v3/v4/v5
RI = []
for v in ('3', '4', '5'):
    with tarfile.open(os.path.join(od0, f'mc_v{v}', f'mc_v{v}_sonuc.tar.gz')) as t:
        for m in t.getmembers():
            if m.name.endswith('.jsonl'):
                d = pd.DataFrame([json.loads(l) for l in t.extractfile(m).read().decode().splitlines() if l.strip()])
                d['src'] = 'v' + v
                RI.append(d)
RI = pd.concat(RI, ignore_index=True)
if 'L' not in RI.columns:
    RI['L'] = 24
RI['L'] = RI['L'].fillna(24)

# dJ mapping from MF_kisitlar.csv
dJ = dict(zip(mf.set, mf.dJ))

# Binder analysis on B7
B7 = R7[R7.kind == 'B']

def pkfit(T, y, npt=2):
    T = np.asarray(T); y = np.asarray(y); j = int(np.nanargmax(y))
    lo, hi = max(j - npt, 0), min(j + npt + 1, len(T))
    if hi - lo < 3:
        return T[j]
    pc = np.polyfit(T[lo:hi], y[lo:hi], 2)
    t = -pc[1] / (2 * pc[0]) if pc[0] < 0 else T[j]
    return t if T[lo] <= t <= T[hi - 1] else T[j]

rows = []
for (mdl, s), g in B7.groupby(['model', 'set']):
    r = dict(model=mdl, set=s)
    for L in (24, 36, 48):
        d = g[g.L == L].sort_values('T')
        r[f'C{L}'] = pkfit(d['T'], d['C'])
        r[f'X{L}'] = pkfit(d['T'], d['chizz'])
        r[f'edge{L}'] = int(np.argmax(d['C'].values)) in (0, len(d) - 1)
    rows.append(r)
PK = pd.DataFrame(rows)
PK['dC_36_48%'] = 100 * (PK.C48 - PK.C36) / PK.C48
PK['dX_36_48%'] = 100 * (PK.X48 - PK.X36) / PK.X48
PK['C_X_48%'] = 100 * (PK.C48 - PK.X48) / PK.C48
PK['sTN'] = PK.apply(lambda r: 0.5 * (max(r.C36, r.C48, r.X36, r.X48) - min(r.C36, r.C48, r.X36, r.X48)), axis=1)
pkd = {(r.model, r.set): r for r in PK.itertuples()}

# Published model mappings
base = {'W4pub': 'Wildes20_r4_x2', 'W2pub': 'Wildes20_r2_x2', 'OlsenD': 'Olsen_U2',
        'YangD': 'Yang_U36', 'AmirDB': 'Amir_U222'}

Z7 = R7[R7.kind == 'Z']
P7 = R7[R7.kind == 'P']

J6 = {'Wildes20_r4_x2': (1.96, 1.96, -0.31, -0.90, 0.0), 'Wildes20_r2_x2': (1.80, 1.80, -0.094, -1.28, 0.0),
      'Ilyas24_x2': (1.12, 2.80, -0.02, -1.06, 0.0, -0.86), 'Lancon_x2': (2.92, 2.92, -0.08, -1.92, 0.0),
      'Yang_U36': (3.13, 3.13, -0.34, -2.01, 0.0), 'Olsen_U2': (2.10, 2.10, -0.21, -2.60, 0.0),
      'Amir_U222': (1.0025, -0.815, -0.31, -1.4275, 0.375), 'own_U3.50': (0.7479, 0.7479, -0.0026, -2.7487, 0.0),
      'own_U3.00': (0.9767, 0.9767, -0.0053, -3.1433, 0.0), 'own_U2.25': (1.5261, 1.5261, 0.0013, -3.9063, 0.0),
      'Chat_U25': (2.60, -2.20, -1.90, -0.60, 0.0)}

rank7 = ['Wildes20_r4_x2', 'Wildes20_r2_x2', 'Lancon_x2', 'Yang_U36', 'Amir_U222', 'Olsen_U2', 'Chat_U25']
lab = {'Wildes20_r4_x2': 'Wildes fit 4 x2 (INS)', 'Wildes20_r2_x2': 'Wildes fit 2 x2 (INS)',
       'Lancon_x2': 'Lancon x2 (INS)', 'Yang_U36': 'Yang U3.6 (DFT+U)',
       'Amir_U222': 'Amirabbasi U2.22 (DFT+U)', 'Olsen_U2': 'Olsen U2 (DFT+U)',
       'Chat_U25': 'Chaturvedi U2.5 (DFT+U)', 'Ilyas24_x2': 'Ilyas x2 (DFT+U)',
       'own_U3.50': 'own U3.50 (20-atom)', 'own_U3.00': 'own U3.00 (20-atom)',
       'own_U2.25': 'own U2.25 (20-atom)'}

sf = spinflip_alanlari.set_index('set')

# Experimental chi from Jena new digitization
ja_path = P('Jena_Fig5c_5T_Hdik_otomatik.csv')
# This is 5T data, need 0.05T for chi
# Load Fig3a digitized data from od0
fa = pd.read_csv(P('Fig3a_Data_0.05T_Dik.csv'), sep=';', decimal=',',
                 header=None, names=['T', 'M'])
fa_clean = fa[fa['T'] > 0]

def rd_new(v):
    return pd.read_csv(v, sep=';', decimal=',', header=None, names=['T', 'y'], encoding='utf-8-sig')

# Joy 1992 data
joy_path = P('Joy_1992_fig3_X_m_vs_K.csv')
ja_new_path = P('new_Fig3a_Data_0.05T_Dik.csv')

joy = rd_new(joy_path)
ja = rd_new(ja_new_path)

# chiMF normalization
Msat_emu_g = 4 * 5585.0 / 183.0  # 4 muB/Fe in emu/g
dE_J_mol = dEexp * 96.485 / 0.183  # meV/Fe -> J/mol/Fe... actually in emu units:
# chiMF = Msat^2 / (2 * dE); need consistent units
# Msat in emu/g, dE in erg/g -> J/kg...
# Let's recompute: Msat = 4 muB/Fe; in emu/mol = 4 * 5585 emu/mol; /183 g/mol = 122.0 emu/g
# dE in emu*T/g units: dEexp meV/Fe * (1e-3 eV * 1.60218e-19 J/eV) / (1.66054e-27 kg / 183) ... 
# dEexp=9.3367 meV/Fe; 1 meV/Fe = 9.3367e-3 * 1.60218e-19 J / (183/6.022e23 * 1e-3 kg) [per kg]
chiMF = 122.1 ** 2 / (2 * 4924.0)
chiMF_mol = chiMF * 1e-4 * 183.0  # for Joy (in emu/mol/Oe units)

rng = np.random.default_rng(1)

def Fgen(T, chi, TN, t1, t2, norm):
    o = np.argsort(T); T = np.asarray(T)[o]; chi = np.asarray(chi)[o]
    return (np.interp(t2 * TN, T, chi) - np.interp(t1 * TN, T, chi)) / norm

def tn_fisher(T, chi, lo=100, hi=135):
    g = pd.Series(chi, index=T).groupby(level=0).mean().sort_index()
    t = g.index.values; y = g.values * t
    tt = np.linspace(lo, hi, 3501); yy = np.interp(tt, t, y); dy = np.gradient(yy, tt)
    return tt[np.argmax(dy)]

sets_exp = {'Jena_eski': (fa_clean['T'].values, fa_clean.M.values / 0.05, chiMF),
            'Jena_yeni': (ja['T'].values, ja.y.values / 0.05, chiMF),
            'Joy1992': (joy['T'].values, joy.y.values, chiMF_mol)}

# Compute F for experimental data
FE_rows = []
for n, (T_, c_, nm) in sets_exp.items():
    r0 = {}
    tf = tn_fisher(T_, c_)
    bsf = [tn_fisher(T_ + rng.normal(0, 0.5, len(T_)), c_ * (1 + rng.normal(0, 0.01, len(c_)))) for _ in range(500)]
    sd = np.std(bsf)
    r0['veri'] = n; r0['TN_Fisher'] = tf; r0['TN_Fisher_bs_sd'] = sd
    for w, (t1, t2) in WIN.items():
        bsv = []
        for _ in range(1000):
            Tj = T_ + rng.normal(0, 0.5, len(T_))
            cj = c_ * (1 + rng.normal(0, 0.01, len(c_)))
            bsv.append(Fgen(Tj, cj, 119, t1, t2, nm))
        r0[f'{w}_bs_sd'] = np.std(bsv)
        for TN in (118, 119, 121, 123, 125):
            r0[f'{w}_TN{TN}'] = Fgen(T_, c_, TN, t1, t2, nm)
        r0[f'{w}_TNFisher'] = Fgen(T_, c_, tf, t1, t2, nm)
    FE_rows.append(r0)
FX = pd.DataFrame(FE_rows)

# Compute T3 (simulation results table) using ana7 functions
T3_rows = []
for s in dJ:
    if s.startswith('experiment'):
        continue
    # Ising
    d = RI[(RI.set == s) & (RI.L == 24)]
    r_pk = pkd[('I', s)]
    if len(d) > 0:
        T3_rows.append(dict(kind='2D', model='Ising', set=s, TN=r_pk.C48, sTN=r_pk.sTN, TN24=r_pk.C24,
                            **case(d, r_pk.C48, dJ[s], r_pk.sTN)))
    # Heis
    d = pd.concat([R6[(R6.kind == 'H') & (R6.set == s)], R7[(R7.kind == 'S') & (R7.set == s)]])
    r_pk = pkd[('H', s)]
    if len(d) > 0:
        T3_rows.append(dict(kind='2D', model='Heis', set=s, TN=r_pk.C48, sTN=r_pk.sTN, TN24=r_pk.C24,
                            **case(d, r_pk.C48, dJ[s], r_pk.sTN)))

for m, b in base.items():
    d = P7[P7.set == m]
    if len(d) == 0:
        continue
    TN = tn_peak(d)
    r_pk = pkd[('H', b)]
    sT = np.hypot(r_pk.sTN, r_pk.C24 - r_pk.C48)
    T3_rows.append(dict(kind='pub', model='Heis+pub', set=m, base=b, D=d.D.iloc[0], TN=TN, sTN=sT, TN24=TN,
                        **case(d, TN, dJ[b], sT)))

# 3D
D3 = pd.concat([R6[R6.kind == '3'].assign(src='v6'), Z7.assign(src='v7')])
D3['Jp'] = D3.Jp.round(4)
for (s, jp), d in D3.groupby(['set', 'Jp']):
    TN = tn_peak(d)
    T3_rows.append(dict(kind='3D', model=f'Ising3D Jp{jp}', set=s, TN=TN, sTN=np.nan, TN24=TN,
                        TN2D24=pkd[('I', s)].C24, **case(d.assign(seed=d.src), TN, dJ[s], 0.0)))

T3 = pd.DataFrame(T3_rows)

# Compute mxall
mxall = {}
for w in WIN:
    vals = []
    for s in rank7:
        for mdl_col in [f'F0_{w}_I', f'F0_{w}_H']:
            sub = T3[(T3.kind == '2D') & (T3.set == s)]
            if len(sub) > 0:
                v = sub.iloc[0].get(mdl_col.replace('_I', '').replace('_H', ''), np.nan)
                # get correct column
        for _, row in T3[(T3.kind == '2D') & (T3.set == s)].iterrows():
            for col in [f'F0_{w}', f'F0_{w}']:
                if col in row.index:
                    vals.append(row[col])
    # simpler approach
    sub_I = T3[(T3.kind == '2D') & (T3.model == 'Ising') & (T3.set.isin(rank7))]
    sub_H = T3[(T3.kind == '2D') & (T3.model == 'Heis') & (T3.set.isin(rank7))]
    sub_P = T3[T3.kind == 'pub']
    all_vals = []
    for sub in (sub_I, sub_H, sub_P):
        col = f'F0_{w}'
        if col in sub.columns:
            all_vals.extend(sub[col].dropna().tolist())
    mxall[w] = max(all_vals) if all_vals else np.nan

# Build TAB (simulation results)
out = []
for s in rank7 + ['Ilyas24_x2', 'own_U3.50', 'own_U3.00', 'own_U2.25']:
    I_rows = T3[(T3.kind == '2D') & (T3.model == 'Ising') & (T3.set == s)]
    H_rows = T3[(T3.kind == '2D') & (T3.model == 'Heis') & (T3.set == s)]
    if len(I_rows) == 0 or len(H_rows) == 0:
        continue
    I = I_rows.iloc[0]; H = H_rows.iloc[0]
    r = dict(set=s, etiket=lab[s],
             siralamada=('evet' if s in rank7 else ('hayir: Ek Bilgi' if s.startswith('Ilyas') else 'hayir: Makale 2')),
             Hamiltonyen='J1-J2-J3(+J4) + D 2.66', TN_Ising_K=I.TN, sTN_Ising_K=I.sTN,
             TN_Heis_K=H.TN, sTN_Heis_K=H.sTN, TN_oran_H_I=H.TN / I.TN,
             spin_norm_1p5xoran=1.5 * H.TN / I.TN, TN_aralik_deneyi_kapsar=bool(H.TN <= 123 and I.TN >= 118))
    for mdl_label, X in (('I', I), ('H', H)):
        for w in WIN:
            r[f'F0_{w}_{mdl_label}'] = X.get(f'F0_{w}', np.nan)
            r[f'F0_{w}_{mdl_label}_se'] = X.get(f'F0_{w}_se', np.nan)
        r[f'F5_ana_{mdl_label}'] = X.get('F5_ana', np.nan)
        r[f'prod5T_ana_{mdl_label}_g2'] = X.get('prod_ana', np.nan)
        r[f'prod5T_ana_{mdl_label}_se'] = X.get('prod_ana_se', np.nan)
    r['dE_meV_Fe'] = sf.loc[s, 'dE_eff_meV']
    r['dE_over_exp'] = r['dE_meV_Fe'] / dEexp
    r['spinflip_T'] = sf.loc[s, 'transitions']
    r['B_direkt_T'] = sf.loc[s, 'B_direct_T']
    r['yarim_plato'] = bool(sf.loc[s, 'plateau_half'])
    r['nseed_H'] = H.get('nseed', np.nan)
    out.append(r)

for m, b in base.items():
    P_rows = T3[(T3.kind == 'pub') & (T3.set == m)]
    if len(P_rows) == 0:
        continue
    P = P_rows.iloc[0]
    r_pk = pkd[('H', b)]
    sT = np.hypot(r_pk.sTN, r_pk.C24 - r_pk.C48)
    r = dict(set=m, etiket=f'{lab[b]} yayimlanmis H', siralamada='evet (yayimlanmis model)',
             Hamiltonyen={'W4pub': 'bilineer + D 2.52 + K1,K2 (S.S)^2', 'W2pub': 'bilineer + D 2.53 + K1 (S.S)^2',
                          'OlsenD': 'bilineer + D 0.101', 'YangD': 'bilineer + D 4.85',
                          'AmirDB': 'bilineer + D 0.2225 + B (DMI yok)'}[m],
             TN_Heis_K=P.TN, sTN_Heis_K=P.sTN, TN_aralik_deneyi_kapsar=bool(118 <= P.TN <= 123))
    for w in WIN:
        r[f'F0_{w}_H'] = P.get(f'F0_{w}', np.nan)
        r[f'F0_{w}_H_se'] = P.get(f'F0_{w}_se', np.nan)
    r['F5_ana_H'] = P.get('F5_ana', np.nan)
    r['prod5T_ana_H_g2'] = P.get('prod_ana', np.nan)
    r['prod5T_ana_H_se'] = P.get('prod_ana_se', np.nan)
    r['dE_meV_Fe'] = sf.loc[b, 'dE_eff_meV']
    r['dE_over_exp'] = r['dE_meV_Fe'] / dEexp
    r['nseed_H'] = P.get('nseed', np.nan)
    out.append(r)

TAB = pd.DataFrame(out)


def dist(lo, hi, a=118, b=123):
    if np.isnan(lo) or np.isnan(hi):
        return np.nan
    if hi < a:
        return hi - a
    if lo > b:
        return lo - b
    return 0.0


TAB['TN_bant_alt_K'] = np.where(TAB.TN_Ising_K.notna(), TAB.TN_Heis_K, TAB.TN_Heis_K)
TAB['TN_bant_ust_K'] = np.where(TAB.TN_Ising_K.notna(), TAB.TN_Ising_K, TAB.TN_Heis_K)
TAB['TN_deneye_uzaklik_K'] = [dist(l, h) for l, h in zip(TAB.TN_bant_alt_K, TAB.TN_bant_ust_K)]
if 'TN_aralik_deneyi_kapsar' in TAB.columns:
    TAB = TAB.drop(columns=['TN_aralik_deneyi_kapsar'])

# Add experimental rows
exp_rows = []
for n, lab_ in (('Jena_yeni', 'Deney: Jena 2026 Fig3a yeniden dijitalestirme (0.05 T, H||c*)'),
                ('Joy1992', 'Deney: Joy & Vasudevan 1992 Fig3 chi_par (tek kristal)')):
    T_, c_, nm = sets_exp[n]
    r0 = FX[FX.veri == n].iloc[0]
    tf, sd = r0.TN_Fisher, r0.TN_Fisher_bs_sd
    r = dict(set=n, etiket=lab_, siralamada='deney',
             TN_bant_alt_K=tf - sd, TN_bant_ust_K=tf + sd,
             TN_Fisher_K=tf, sTN_Fisher_K=sd)
    for w, (t1, t2) in WIN.items():
        vals = [Fgen(T_, c_, t, t1, t2, nm) for t in (tf - sd, tf, tf + sd)]
        bs_sd = r0[f'{w}_bs_sd']
        r[f'F_exp_{w}'] = vals[1]
        r[f'F_exp_{w}_min'] = min(vals) - bs_sd
        r[f'F_exp_{w}_max'] = max(vals) + bs_sd
        r[f'F_exp_{w}_over_Fmodel_max'] = vals[1] / mxall[w] if mxall[w] > 0 else np.nan
    r.update(dE_meV_Fe=dEexp, spinflip_T='[(39.35, 0, 0.5), (41.3, 0.5, 1.0)] Wildes 2020',
             yarim_plato=True,
             normalizasyon='chi_MF = Msat^2/(2 dE), Msat 4 muB/Fe, dE 9.3367 meV/Fe (Wildes 2020); 0.027703 emu/mol/Oe')
    exp_rows.append(r)

# Remove old deney row if present
TAB2 = pd.concat([TAB[~TAB.set.isin(['deney'])], pd.DataFrame(exp_rows)], ignore_index=True)
TAB2.to_csv('Tablo_uc_kisit.csv', index=False)