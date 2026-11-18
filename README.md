# FePS3-thermodynamic-constraints

Data and code for

> N. Paran, *Three thermodynamic constraints on exchange parameters of FePS₃: Néel temperature, longitudinal-susceptibility step and spin-flip fields*, submitted to Physical Review B.

The article tests seven published exchange-parameter sets for bulk FePS₃ against three thermodynamic constraints: the Néel temperature, the longitudinal-susceptibility step F = [J(**Q**) − J(0)] Δχ∥, and the zigzag → ferromagnetic energy E_FM − E_zz inferred from the 39–41 T spin-flip steps. Every number in the tables and figures is either stored in this repository or produced from it by the scripts listed below. `analysis/REPRODUCTION_CHECK.md` records which outputs were regenerated and how closely they agree with the stored files.

The Monte Carlo campaigns were run with Python 3 and NumPy on the TRUBA cluster (TÜBİTAK ULAKBİM). The analysis and figure scripts need Python ≥ 3.10 with NumPy, SciPy, pandas and Matplotlib (`requirements.txt` lists the versions used for the reproduction check). No DFT calculation is part of this article.

## Layout

| Folder | Contents |
|---|---|
| `mc/lib/` | Monte Carlo engines shared by the campaigns: `fepslib.py` (Blume–Capel engine with J₁–J₄, used by v3 and v4) and `fepslib2.py` (adds a separate interchain J₂, used from v5 on). |
| `mc/mc_v3/` … `mc/mc_v7/` | One folder per production campaign: driver (`run_mcN.py`), SLURM script (`run_mcN.sh`), summary script (`ozetN.py`) and the compressed raw output `mc_vN_sonuc.tar.gz` (contains `mcN_raw.jsonl`, logs and summaries). v6 adds `fepslib3.py` (classical Heisenberg + D, generalized Blume–Capel), v7 adds `fepslib4.py`. |
| `data/experiment/` | Digitized experimental curves (see *Experimental data* below). |
| `data/derived/` | Tables derived from the raw Monte Carlo output and the experimental curves; these are the files the article quotes. |
| `analysis/` | Scripts that produce the derived tables and the figures, plus `REPRODUCTION_CHECK.md`. |
| `figures/` | The figures as submitted (PNG and PDF). |
| `MANIFEST.txt` | MD5 checksums of all files. |

### Monte Carlo campaigns (SM Table S2)

| Campaign (SM) | Folder | Model | Cost (core-h) |
|---|---|---|---|
| A | `mc/mc_v3` | Blume–Capel, five literature sets and two test sets, L = 24 | 150 |
| B | `mc/mc_v4` | Blume–Capel size test, L = 16, 24, 36 | 64.7 |
| C | `mc/mc_v5` | Blume–Capel, Wildes fits 2 and 4 and the Ilyas set | 57.4 |
| D | `mc/mc_v6` | Heisenberg + D for all sets; four-layer Blume–Capel stacks | 176.8 |
| E | `mc/mc_v7` | T_N for both models, published Hamiltonians, L = 24/36/48, extra seeds | 164.6 |

To rerun a campaign, copy the library files it imports into its folder (v3, v4: `mc/lib/fepslib.py`; v5: `fepslib2.py`, already present; v6: `fepslib3.py` + `mc/lib/fepslib2.py`; v7: `fepslib4.py` + `mc/mc_v6/fepslib3.py` + `mc/lib/fepslib2.py`) and submit `run_mcN.sh` with `sbatch`. `mc/mc_v5/yama_fepslib2.py` documents how `fepslib2.py` was derived from `fepslib.py`.

The raw files also contain three exchange sets labelled `own_U2.25`, `own_U3.00` and `own_U3.50`. They come from an earlier 20-atom DFT+U energy mapping of the author and are **not used in the article** (the 20-atom cell aliases some of the exchange paths). They are left in the raw output, and some derived tables list them as extra rows, so that the files are identical to what was produced; no table or figure of the article uses them.

## Running the analysis

Each script can be run from any working directory. It finds the repository through its own location, unpacks the raw Monte Carlo archives it needs into the current directory and writes its output there:

```
MPLBACKEND=Agg python analysis/table_three_constraints.py
```

## Where each result comes from

| Article item | File(s) | Produced by |
|---|---|---|
| Table I (exchange sets) | set dictionaries in `mc/mc_v*/run_mc*.py` and in the analysis scripts | conversions in SM Table S1 |
| Table II, SM Table S5 (three constraints) | `data/derived/Tablo_uc_kisit.csv` | `analysis/table_three_constraints.py` (raw MC v3–v7 + `MF_kisitlar.csv`, `spinflip_alanlari.csv`, `LSWT_kontrol.csv`, `MCE_5T_deney_karsilastirma.csv`) |
| Fig. 1 (exchange paths and zigzag order) | `figures/Sekil_J_orgu.*` | `analysis/fig1_exchange_paths.py` (geometry of Jena et al. 2026, Table I) |
| Fig. 2 (normalized χ∥ vs T/T_N) | `figures/Sekil_chi_normal.*` | `analysis/fig2_chi_normalized.py` (raw MC v7, digitized χ∥ of Jena et al. and Joy & Vasudevan) |
| Fig. 3 (three constraints) | `figures/Sekil_uc_kisit.*` | `analysis/fig3_three_constraints.py` → `analysis/sekil_uc_kisit.py` |
| Fig. 4 (spin-flip fields, LSWT bands) | `figures/Sekil_spinflip_A.*` | `analysis/fig4_spinflip_lswt.py` (`spinflip_alanlari.csv`, `LSWT_kontrol.csv`) |
| Spin-flip fields (Sec. III, Fig. 4a) | `data/derived/spinflip_alanlari.csv` | `analysis/spinflip_fields.py` |
| LSWT band check (Fig. 4b) | `data/derived/LSWT_kontrol.csv` | `analysis/lswt_check.py` |
| Mean-field T_N and F (SM Table S8) | `data/derived/MF_kisitlar.csv` | `analysis/mf_constraints.py` (writes `MF_kisitlar_regen.csv`, see `REPRODUCTION_CHECK.md`) |
| Experimental F in three windows (SM Table S4) | `data/derived/F_deney_pencereler.csv` | `analysis/exp_F_windows.py` |
| dχ/dT-based T_N windows for the Jena et al. curve | `data/derived/Jena_dchidT_pencere.csv` | stored intermediate (not regenerated by a script here) |
| C and χ_zz peak positions vs L (SM Table S3) | `data/derived/TN_tepe_L24_36_48.csv` | `analysis/tn_peaks.py` |
| T_N from Fisher's relation vs C peak | `data/derived/TN_Fisher_vs_C_MC.csv` | `analysis/tn_fisher_vs_C.py` |
| Per-set MC state summary (campaign E) | `data/derived/MC_v7_durumlar.csv` | `analysis/mc_v7_states.py` |
| Heisenberg + D and four-layer stacks (SM Table S6) | `data/derived/MC_v6_analiz.csv` | summary of `mc/mc_v6` raw output (stored; not regenerated by a script here) |
| 5 T magnetocaloric comparison | `data/derived/MCE_5T_deney_karsilastirma.csv` | stored intermediate (not regenerated by a script here) |

`analysis/ana7.py` is the original analysis module that reads the raw MC v3–v7 output (C(T) and χ peaks, T_N and its uncertainty, F from the zero-field and 5 T data); `table_three_constraints.py` builds on it. `analysis/figstyle.py` sets the figure style.

## Experimental data

The files in `data/experiment/` are digitizations made by the author from published figures; the published figures remain the primary source.

| File | Source |
|---|---|
| `Joy_1992_fig3_X_m_vs_K.csv/.json` | P. A. Joy and S. Vasudevan, Phys. Rev. B **46**, 5425 (1992), Fig. 3 (molar susceptibility) |
| `new_Fig3a_Data_0.05T_Dik.csv/.json`, `new_Fig3b_Data__0.05T_Paralel.csv`, `new_Fig3b_Data_0.05T_Dik.json` | M. Jena *et al.*, Phys. Rev. B **113**, 064429 (2026), Fig. 3 (0.05 T, H ⊥ and ∥ to the layers) |
| `Fig3a_Data_0.05T_Dik.csv` | earlier digitization of the same panel, kept for comparison |
| `Jena_Fig5c_5T_Hdik_otomatik.csv` | Jena *et al.* (2026), Fig. 5c (5 T magnetocaloric curve, automatic digitization) |

The spin-flip fields (39.35 T and 41.3 T) are taken from A. R. Wildes *et al.*, Phys. Rev. B **101**, 024415 (2020).

## Conventions

H = −Σ_⟨ij⟩ J_ij **S**_i·**S**_j − D Σ_i (S_i^z)², each bond counted once, S = 2, J > 0 ferromagnetic, energies in meV. Published sets written with other conventions are converted as listed in SM Table S1 (for example, Lançon *et al.* and Wildes *et al.* count each bond twice, so their J are multiplied by 2). File and column names partly use Turkish words: *deney* = experiment, *kısıt* = constraint, *pencere* = window, *durum* = state, *tepe* = peak, *alan* = field, *şekil* = figure, *Dik/Paralel* = perpendicular/parallel.

## License

MIT (see `LICENSE`).
