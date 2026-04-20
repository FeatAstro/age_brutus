# orion-ob1 — Star formation history of the local Milky Way

M1 internship project at LIRA (2026), supervised by Alexis Quintana.

**Goal**: Extend OB association membership catalogs to lower-mass, cooler stars
in the Orion OB1 complex, combining 5D kinematic clustering with Bayesian
isochrone fitting to derive ages, distances, extinctions, and metallicities
for stellar groups within ~1 kpc of the Sun.

Methodology follows Quintana et al. (2025) and Hernández et al. (2023).

---

## Repository layout

```
orion-ob1/
├── Cat/
│   ├── raw/          # ADQL query outputs — gitignored, do not edit
│   └── processed/    # after ZP corrections and cross-matches
├── Py/
│   ├── 01_query_gaia.py       # Gaia DR3 download + 2MASS/PanSTARRS joins
│   ├── 02_hdbscan.py          # 5D kinematic clustering
│   ├── 03_fitting.py          # Bayesian isochrone fitting (brutus + dynesty)
│   ├── 03_map.py              # cross-match clusters with region labels
│   ├── config.py              # PARAMS dicts per target region
│   └── utils/
│       ├── photometry.py      # ZP corrections, extinction, CMD cuts
│       ├── clustering.py      # HDBSCAN wrappers and 5D scaler
│       └── brutus_helpers.py  # worker_init, live point pre-generation
└── brutus/           # git submodule — Speagle et al. isochrone code
```

---

## Setup

```bash
# 1. Clone with submodule
git clone --recurse-submodules https://github.com/YOUR_USERNAME/orion-ob1.git
cd orion-ob1

# 2. Create conda environment
conda env create -f environment.yml
conda activate orion-ob1

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
python Py/01_query_gaia.py --region OBP-a --test
python Py/01_query_gaia.py --region OBP-a
```
Downloads Gaia DR3 cross-matched with 2MASS, PanSTARRS, and BailerJones
distances. Applies Lindegren+2021 parallax ZP and SPICOR corrections.
Output: `Cat/raw/{region}_raw.fits`

### Step 2 — Kinematic membership
```bash
python Py/02_hdbscan.py --region OBP-a
```
Applies selection cuts (parallax range, σ_ϖ/ϖ, RUWE, CMD), runs HDBSCAN
in 5D space (l, b, ϖ, μ_α*, μ_δ), and assigns membership probabilities.
Output: `Cat/processed/{region}_members.fits`

### Step 3 — Isochrone fitting
```bash
python Py/03_fitting.py --region OBP-a
python Py/03_fitting.py --region OBP-a --resume   # continue interrupted run
```
Bayesian fitting with brutus (MIST models, EEP ≥ 202) and dynesty nested
sampling. **Note**: MIST excludes pre-main-sequence stars; only use on
regions older than ~5–10 Myr (Orion OB1 sub-regions, not the ONC).
Output: `Cat/processed/{region}/{cluster_id}/`

### Step 4 — Region map
```bash
python Py/03_map.py --region OBP-a
```
Cross-matches HDBSCAN clusters with Sanchez-Sanjuan 2024 region labels
and produces sky plots.

---

## Key parameters (see `Py/config.py`)

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
  independently via `worker_init` (see `Py/utils/brutus_helpers.py`)

---

## References

- Quintana et al. (2025)
- Hernández et al. (2023)
- Sanchez-Sanjuan (2024) — region labels
- Lindegren et al. (2021) — Gaia DR3 parallax zero-point
- Maíz Apellániz (2022) — SPICOR correction
- Rybizki et al. (2022) — Gaia fidelity scores
- Güver & Özel (2009) — N_H / A(V) relation
