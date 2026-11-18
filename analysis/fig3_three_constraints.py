"""fig3_three_constraints.py - Fig. 3 (three constraints) from data/derived/Tablo_uc_kisit.csv and MF_kisitlar.csv."""
import os, sys
import pandas as pd, matplotlib
matplotlib.use('Agg')
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'analysis'))
from figstyle import apply_figure_style, panel_letter
from sekil_uc_kisit import make_figure
TAB = pd.read_csv(os.path.join(REPO, 'data', 'derived', 'Tablo_uc_kisit.csv'))
MF = pd.read_csv(os.path.join(REPO, 'data', 'derived', 'MF_kisitlar.csv'))
make_figure(TAB, MF, apply_figure_style, panel_letter, out='Sekil_uc_kisit')
