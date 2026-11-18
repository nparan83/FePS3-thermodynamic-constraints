"""yama_fepslib2.py - fepslib v1 -> v2: J2 bagini J2par (dj = 0, zincir ici) ve J2perp (dj != 0) olarak ayirir. 5 elemanli J eskisi gibi (J2perp = J2).
Kullanim: python3 yama_fepslib2.py ../mc_v1/fepslib.py fepslib2.py . ASCII-only."""
import sys
s = open(sys.argv[1]).read()
rep = [("LABS = ('J1par', 'J1perp', 'J2', 'J3', 'J4')", "LABS = ('J1par', 'J1perp', 'J2', 'J3', 'J4', 'J2perp')"),
       ("elif abs(d - S3) < 1e-9: lab = 'J2'", "elif abs(d - S3) < 1e-9: lab = 'J2' if dj == 0 else 'J2perp'"),
       ("    J = list(J) + [0.0] * (5 - len(J))\n", "    J = list(J) + [0.0] * (5 - len(J))\n    J = J + [J[2]] if len(J) == 5 else J\n")]
for a, b in rep:
    assert s.count(a) == 1, a
    s = s.replace(a, b)
open(sys.argv[2], 'w').write(s.replace('fepslib.py v1', 'fepslib2.py v2 (J2par/J2perp)', 1))
print('tamam')
