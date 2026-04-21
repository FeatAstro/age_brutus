# Star formation history of the local Milky Way

M1 internship project at LIRA (2026), supervised by Alexis Quintana.

**Goal**: Extend OB association membership catalogs to lower-mass, cooler stars
in the Orion OB1 complex, combining 5D kinematic clustering with Bayesian
isochrone fitting to derive ages, distances, extinctions, and metallicities
for stellar groups within ~1 kpc of the Sun, using Brutus algorithm.

Methodology follows Quintana et al. (2025).

---

## Repository layout

```
age_brutus/
├── Download_Gaia.py     # Gaia DR3 download + 2MASS/PanSTARRS joins
├── HDBSCAN_Gaia.py      # 5D kinematic clustering
├── Brutus_Gaia.py       # Bayesian isochrone fitting (brutus + dynesty)
├── Map_Gaia.py          # cross-match clusters with region labels
├── data/
│   ├── raw/             # ADQL query outputs — gitignored, do not edit
│   └── processed/       # after ZP corrections and cross-matches
├── outputs/             # per-cluster dynesty results — gitignored
├── images/
│   ├── HDBSCAN/         # membership and kinematic plots
│   └── compare/         # isochrone fits and comparison plots
└── brutus/              # git submodule — Speagle et al. isochrone code
```

---

## Setup

```bash
# 1. Clone with submodule
git clone --recurse-submodules https://github.com/FeatAstro/age_brutus.git
cd age_brutus

# 2. Create conda environment
conda env create -f environment.yml
conda activate age_brutus

# 3. Install brutus in editable mode
pip install -e ./brutus
```

If you already cloned without `--recurse-submodules`:
```bash
git submodule update --init --recursive
```

---

## Pipeline

### Step 1 — Download photometry
```bash
python Download_Gaia.py --test
python Download_Gaia.py 
```
Downloads Gaia DR3 cross-matched with 2MASS, PanSTARRS, and BailerJones
distances. Applies Lindegren+2021 parallax ZP and SPICOR corrections.
Output: `data/raw/{region}_raw.fits`
Note: Update parameters for each complex (ra, dec, ...)

### Step 2 — Kinematic membership
```bash
python HDBSCAN_Gaia.py 
```
Applies selection cuts (parallax range, σ_ϖ/ϖ, RUWE, CMD), runs HDBSCAN
in 5D space (l, b, ϖ, μ_α*, μ_δ), and assigns membership probabilities.
Output: `data/processed/{region}_members.fits`
Note: Update parameters

### Step 3 — Isochrone fitting
```bash
python Brutus_Gaia.py 
python Brutus_Gaia.py --resume   # continue interrupted run
```
Bayesian fitting with brutus (MIST models, EEP ≥ 202) and dynesty nested
sampling. **Note**: MIST excludes pre-main-sequence stars; only use on
regions older than ~5–10 Myr (not the ONC for instance).
Output: `outputs/{region}/{cluster_id}/`
Note: Update parameters

### Step 4 — Region map
```bash
python Map_Gaia.py 
```
Cross-matches HDBSCAN clusters with Sanchez-Sanjuan 2024 region labels
and produces sky plots. Output: `images/hdbscan/` and `images/compare/`

---

## Key parameters

| Parameter | Typical state | Notes |
|-----------|--------------|-------|
| `log_age` | free | main target |
| `feh` | free or fixed at 0.0 | depends on region S/N |
| `dist` | free | degeneracy with A(V) |
| `Av` | free | A(V)–distance degeneracy handled by dynesty |
| `f_field` | fixed at 0.05 | HDBSCAN pre-cleans membership |

---

## Notes on brutus / MIST

- MIST stellar models cover EEP ≥ 202 (ZAMS and above)
- Stars below ~3–4 M☉ on the pre-main sequence are excluded
- For clusters younger than ~5–10 Myr the likelihood becomes flat
  and the outlier model dominates — this is a model constraint, not a bug
- `StellarPop` objects cannot be pickled; workers load `stellarpop`
  independently via `worker_init`

---

## References

- Quintana et al. (2025)
- Speagle et al. (2024) — brutus isochrone fitting code https://github.com/joshspeagle/brutus
- Speagle (2020) — dynesty nested sampling
- Koposov et al. — dynesty (current maintainers)
- Kroupa (2001) — IMF used in MIST/brutus
- Choi et al. (2016) — MIST stellar models
- Dotter (2016) — MIST/EEP framework
- Hernández et al. (2023) 
- Sanchez-Sanjuan (2024) — region labels
- Lindegren et al. (2021) — Gaia DR3 parallax zero-point
- Maíz Apellániz (2022) — SPICOR correction
- Rybizki et al. (2022) — Gaia fidelity scores
- Güver & Özel (2009) — N_H / A(V) relation
