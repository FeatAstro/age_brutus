#!/usr/bin/env python3
"""
Script 1 — Download from Gaia archive and apply parallax correction
===================================================================
Queries Gaia DR3 + BailerJones distances for a RA/Dec rectangular box,
cross-matches 2MASS and PanSTARRS via the DR3 neighbour tables,
applies the Lindegren+2021 zero-point and Maiz Apellaniz 2022
uncertainty correction, then saves a single merged FITS file.

Usage:
    python Download_Gaia.py                        # run everything
    python Download_Gaia.py --skip-gaia            # skip step 1
    python Download_Gaia.py --skip-bj              # skip step 3
    python Download_Gaia.py --skip-tmass           # skip step 4
    python Download_Gaia.py --skip-ps              # skip step 5
    python Download_Gaia.py --skip-fidelity        # skip step 6
    python Download_Gaia.py --skip-gaia --skip-bj  # combine any flags

Output : <path_data_processed>/catalog_complete_<name_complex>.fits
"""

import os
import sys
import numpy as np
import warnings
from astropy.table import Table, join, unique
from astropy.utils.exceptions import AstropyWarning
from astroquery.gaia import Gaia
from scipy.interpolate import CubicSpline

warnings.simplefilter('ignore', AstropyWarning)
Gaia.ROW_LIMIT = -1

# ----------- Paths 
path_data_raw       = 'data/raw/'
path_data_processed = 'data/processed/'
name_complex 		= 'Orion_OB1'
os.makedirs(path_data_raw, exist_ok=True)
os.makedirs(path_data_processed, exist_ok=True)

# ----------- Parameters
# Sky box and distance range — specific to a given complex
RA_MIN  =  75.0
RA_MAX  =  90.0
DEC_MIN = -14.0
DEC_MAX =  16.0

PLX_MIN 	= 2.0   # mas — Orion OB1 range
PLX_MAX 	= 3.6   # mas 
PLX_SNR_MIN = 20.0

MAX_G       =   19.0

# ----------- Skip flags
SKIP_GAIA     = '--skip-gaia'     in sys.argv
SKIP_BJ       = '--skip-bj'       in sys.argv
SKIP_TMASS    = '--skip-tmass'    in sys.argv
SKIP_PS       = '--skip-ps'       in sys.argv
SKIP_FIDELITY = '--skip-fidelity' in sys.argv

print("Skip flags:", flush=True)
print(f"  gaia={SKIP_GAIA}  bj={SKIP_BJ}  tmass={SKIP_TMASS}  ps={SKIP_PS}  fidelity={SKIP_FIDELITY}", flush=True)

# ----------- Helper
def query(q):
    t = Gaia.launch_job_async(q).get_results()
    for col in t.colnames:
        t.rename_column(col, col.lower())
    return t


# ==============================================================================
# 0. Test
# ==============================================================================
#job = Gaia.launch_job_async("""
#SELECT TOP 10 source_id, ra, dec, parallax
#FROM gaiadr3.gaia_source
#WHERE ra BETWEEN 83 AND 84
#AND dec BETWEEN -6 AND -5
#""")
#print(job.get_results())

# ==============================================================================
# 1. Gaia DR3
# ==============================================================================
if SKIP_GAIA:
    print("Step 1: loading Gaia from file ...", flush=True)
    gaia_t = Table.read(path_data_raw + 'raw_gaia.fits')
    # Example of changes we can make with an existing fits file
    #for col in gaia_t.colnames:
    #	gaia_t.rename_column(col, col.lower())
else:
    gaia_query = f"""
    SELECT
        g.source_id,
        g.ra, g.ra_error, g.dec, g.dec_error, g.l, g.b,
        g.parallax, g.parallax_error,
        g.pmra, g.pmra_error, g.pmdec, g.pmdec_error,
        g.radial_velocity, g.radial_velocity_error,
        g.phot_g_mean_mag, g.phot_bp_mean_mag, g.phot_rp_mean_mag,
        g.phot_g_mean_flux_over_error,
        g.phot_bp_mean_flux_over_error,
        g.phot_rp_mean_flux_over_error,
        g.nu_eff_used_in_astrometry, g.pseudocolour, g.ecl_lat,
        g.ruwe, g.astrometric_params_solved
    FROM gaiadr3.gaia_source AS g
    WHERE g.ra  BETWEEN {RA_MIN}  AND {RA_MAX}
      AND g.dec BETWEEN {DEC_MIN} AND {DEC_MAX}
      AND g.phot_g_mean_mag     < {MAX_G}
      AND g.parallax            > {PLX_MIN}
      AND g.parallax            < {PLX_MAX}
      AND g.parallax_over_error > {PLX_SNR_MIN}
    """
    print("Step 1: querying Gaia DR3 ...", flush=True)
    gaia_t = query(gaia_query)
    print(f"  {len(gaia_t)} sources")
    gaia_t.write(path_data_raw + 'raw_gaia.fits', format='fits', overwrite=True)

# ==============================================================================
# 2. BailerJones — sky box query, no upload needed
# ==============================================================================
if SKIP_BJ:
    print("Step 2: loading BailerJones from file ...", flush=True)
    bj_t = Table.read(path_data_raw + 'raw_bailerjones.fits')
else:
    bj_query = f"""
	SELECT 
		d.source_id, d.r_med_geo, d.r_lo_geo, d.r_hi_geo
		FROM external.gaiaedr3_distance AS d
		INNER JOIN gaiadr3.gaia_source AS g ON d.source_id = g.source_id
		WHERE g.ra  BETWEEN {RA_MIN} AND {RA_MAX}
		  AND g.dec BETWEEN {DEC_MIN} AND {DEC_MAX}
		  AND g.phot_g_mean_mag     < {MAX_G}
		  AND g.parallax            > {PLX_MIN}
		  AND g.parallax            < {PLX_MAX}
		  AND g.parallax_over_error > {PLX_SNR_MIN}
		  AND d.r_med_geo > 0
		"""
    print("Step 2: querying BailerJones ...", flush=True)
    bj_t = query(bj_query)
    print(f"  {len(bj_t)} rows")
    bj_t.write(path_data_raw + 'raw_bailerjones.fits', format='fits', overwrite=True)

# ==============================================================================
# 3. 2MASS — sky box query
# ==============================================================================
if SKIP_TMASS:
    print("Step 3: loading 2MASS from file ...", flush=True)
    tmass_t = Table.read(path_data_raw + 'raw_2mass.fits')
else:
    tmass_query = f"""
    SELECT bn.source_id AS dr3_source_id,
           tmass.j_m, tmass.j_msigcom,
           tmass.h_m, tmass.h_msigcom,
           tmass.ks_m, tmass.ks_msigcom, tmass.ph_qual
    FROM gaiadr3.gaia_source AS g
    INNER JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS bn ON g.source_id = bn.source_id
    INNER JOIN gaiadr3.tmass_psc_xsc_join AS xj ON bn.clean_tmass_psc_xsc_oid = xj.clean_tmass_psc_xsc_oid
    INNER JOIN gaiadr2.tmass_original_valid AS tmass ON xj.original_psc_source_id = tmass.designation
    WHERE g.ra  BETWEEN {RA_MIN} AND {RA_MAX}
      AND g.dec BETWEEN {DEC_MIN} AND {DEC_MAX}
      AND g.phot_g_mean_mag     < {MAX_G}
      AND g.parallax            > {PLX_MIN}
      AND g.parallax 			< {PLX_MAX}
      AND g.parallax_over_error > {PLX_SNR_MIN}
    """
    print("Step 3: querying 2MASS ...", flush=True)
    tmass_t = query(tmass_query)
    print(f"  {len(tmass_t)} rows")
    tmass_t.write(path_data_raw + 'raw_2mass.fits', format='fits', overwrite=True)

# ==============================================================================
# 4. PanSTARRS — sky box query
# ==============================================================================
if SKIP_PS:
    print("Step 4: loading PanSTARRS from file ...", flush=True)
    ps_t = Table.read(path_data_raw + 'raw_panstarrs.fits')
else:
    ps_query = f"""
    SELECT bn.source_id AS dr3_source_id,
           ps.g_mean_psf_mag, ps.g_mean_psf_mag_error,
           ps.r_mean_psf_mag, ps.r_mean_psf_mag_error,
           ps.i_mean_psf_mag, ps.i_mean_psf_mag_error,
           ps.z_mean_psf_mag, ps.z_mean_psf_mag_error,
           ps.y_mean_psf_mag, ps.y_mean_psf_mag_error
    FROM gaiadr3.gaia_source AS g
    INNER JOIN gaiadr3.panstarrs1_best_neighbour AS bn ON g.source_id = bn.source_id
    INNER JOIN external.panstarrs1_original_valid AS ps ON bn.original_ext_source_id = ps.obj_id
    WHERE g.ra  BETWEEN {RA_MIN} AND {RA_MAX}
      AND g.dec BETWEEN {DEC_MIN} AND {DEC_MAX}
      AND g.phot_g_mean_mag     < {MAX_G}
      AND g.parallax            > {PLX_MIN}
      AND g.parallax 			< {PLX_MAX}
      AND g.parallax_over_error > {PLX_SNR_MIN}
    """
    print("Step 4: querying PanSTARRS ...", flush=True)
    ps_t = query(ps_query)
    print(f"  {len(ps_t)} rows")
    ps_t.write(path_data_raw + 'raw_panstarrs.fits', format='fits', overwrite=True)

# ==============================================================================
# 5. Fidelity — sky box query
# ==============================================================================
if SKIP_FIDELITY:
    print("Step 5: loading fidelity from file ...", flush=True)
    try:
        fidelity_t = Table.read(path_data_raw + 'raw_fidelity.fits')
    except FileNotFoundError:
        print("  Fidelity file not found, skipping.", flush=True)
        fidelity_t = None
else:
    fidelity_query = f"""
    SELECT g.source_id, f.fidelity_v2 AS fidelity
    FROM gaiadr3.gaia_source AS g
    INNER JOIN external.gaiadr3_spurious AS f ON g.source_id = f.source_id
    WHERE g.ra  BETWEEN {RA_MIN} AND {RA_MAX}
      AND g.dec BETWEEN {DEC_MIN} AND {DEC_MAX}
      AND g.phot_g_mean_mag     < {MAX_G}
      AND g.parallax            > {PLX_MIN}
      AND g.parallax 			< {PLX_MAX}
      AND g.parallax_over_error > {PLX_SNR_MIN}
    """
    print("Step 5: querying fidelity ...", flush=True)
    try:
        fidelity_t = query(fidelity_query)
        print(f"  {len(fidelity_t)} rows")
        fidelity_t.write(path_data_raw + 'raw_fidelity.fits', format='fits', overwrite=True)
    except Exception as e:
        print(f"  Fidelity query failed ({e}), continuing without it.")
        fidelity_t = None
        
# ==============================================================================
# 6. Merge
# ==============================================================================
print("Step 6: merging ...", flush=True)
merged = join(gaia_t, bj_t, keys='source_id', join_type='inner')
print(f"  After BailerJones distance cut: {len(merged)}")
tmass_clean = unique(tmass_t, keys='dr3_source_id')
merged = join(merged, tmass_clean, keys_left='source_id', keys_right='dr3_source_id', join_type='left')
merged = join(merged, ps_t,        keys_left='source_id', keys_right='dr3_source_id', join_type='left')
if fidelity_t is not None:
    merged = join(merged, fidelity_t, keys='source_id', join_type='left')
    n_fid = np.sum(np.isfinite(np.array(merged['fidelity'], dtype=float)))
    print(f"  Fidelity matched: {n_fid}")
print(f"  Total: {len(merged)}"
      f"  |  2MASS: {np.sum(np.isfinite(np.array(merged['j_m'], dtype=float)))}"
      f"  |  PS: {np.sum(np.isfinite(np.array(merged['g_mean_psf_mag'], dtype=float)))}")

# ==============================================================================
# 7. Parallax zero-point correction (Lindegren+2021 + Maiz Apellaniz 2022)
# ==============================================================================
def ZPEDR3(gmag, nueff, lat, npar):
    gcut = np.array([6.0,10.8,11.2,11.8,12.2,12.9,13.1,15.9,16.1,17.5,19.0,20.0,21.0])
    q500 = np.array([-26.98,-27.23,-30.33,-33.54,-13.65,-19.53,-37.99,-38.33,-31.05,-29.18,-18.40,-12.65,-18.22])
    q501 = np.array([ -9.62, -3.07, -9.23,-10.08,  -0.07, -1.64, +2.63, +5.61, +2.83, -0.09, +5.98, -4.57,-15.24])
    q502 = np.array([+27.40,+23.04, +9.08,+13.28,  +9.35,+15.86,+16.14,+15.42, +8.59, +2.41, -6.46, -7.46,-18.54])
    q510 = np.array([-25.1, +35.3, -88.4,-126.7,-111.4, -66.8,  -5.7,     0,     0,     0,     0,     0,     0])
    q511 = np.array([ -0.0, +15.7, -11.8, +11.6, +40.6, +20.6, +14.0, +18.7, +15.5, +24.5,  +5.5, +97.9,+128.2])
    q520 = np.array([-1257.,-1257.,-1257.,-1257.,-1257.,-1257.,-1257.,-1189.,-1404.,-1165.,    0,     0,     0])
    q530 = np.array([    0,     0,     0,     0,     0,     0,+107.9,+243.8,+105.5,+189.7,    0,     0,     0])
    q540 = np.array([    0,     0,     0,     0,     0,     0,+104.3,+155.2,+170.7,+325.0,+276.6,    0,     0])
    q600 = np.array([-27.85,-28.91,-26.72,-29.04,-12.39,-18.99,-38.29,-36.83,-28.37,-24.68,-15.32,-13.73,-29.53])
    q601 = np.array([ -7.78, -3.57, -8.74, -9.69, -2.16, -1.93, +2.59, +4.20, +1.99, -1.37, +4.01,-10.92,-20.34])
    q602 = np.array([+27.47,+22.92, +9.36,+13.63,+10.23,+15.90,+16.20,+15.76, +9.28, +3.52, -6.03, -8.30,-18.74])
    q610 = np.array([-32.1,  +7.7, -30.3, -49.4, -92.6, -57.2, -10.5, +22.3, +50.4, +86.8, +29.2, -74.4, -39.5])
    q611 = np.array([+14.4, +12.6,  +5.6, +36.3, +19.8,  -8.0,  +1.4, +11.1, +17.2, +19.8, +14.1,+196.4,+326.8])
    q612 = np.array([ +9.5,  +1.6, +17.2, +17.7, +27.6, +19.9,  +0.4, +10.0, +13.7, +21.3,  +0.4, -42.0,-262.3])
    q620 = np.array([ -67., -572.,-1104.,-1129., -365., -554., -960.,-1367.,-1351.,-1380., -563., +536.,+1598.])
    n=len(gmag); b0=np.ones(n); b1=np.sin(np.deg2rad(lat)); b2=b1**2-1./3.
    c1=np.zeros(n); c2=np.zeros(n); c3=np.zeros(n); c4=np.zeros(n)
    for i in range(n):
        nu=nueff[i]
        if   nu<=1.24:              c1[i]=-0.24; c2[i]=0.24**3; c3[i]=nu-1.24
        elif nu<=1.72:
            c1[i]=nu-1.48
            if nu<=1.48: c2[i]=(1.48-nu)**3
        else:                       c1[i]=0.24; c4[i]=nu-1.72
    qi=lambda g,q: np.interp(g,gcut,q)
    z5=(qi(gmag,q500)*b0+qi(gmag,q501)*b1+qi(gmag,q502)*b2+qi(gmag,q510)*c1*b0+qi(gmag,q511)*c1*b1+qi(gmag,q520)*c2*b0+qi(gmag,q530)*c3*b0+qi(gmag,q540)*c4*b0)
    z6=(qi(gmag,q600)*b0+qi(gmag,q601)*b1+qi(gmag,q602)*b2+qi(gmag,q610)*c1*b0+qi(gmag,q611)*c1*b1+qi(gmag,q612)*c1*b2+qi(gmag,q620)*c2*b0)
    return [z5[j] if npar[j]==5 else z6[j] for j in range(n)]

def SPICOR(spi, gmag, ruwe, npar):
    gref=np.array([6.50,7.50,8.50,9.50,10.25,10.75,11.25,11.75,12.25,12.75,13.25,13.75,14.25,14.75,15.25,15.75,16.25,16.75,17.25,17.75])
    kref=np.array([2.62,2.38,2.06,1.66,1.22,1.41,1.76,1.76,1.90,1.92,1.61,1.50,1.39,1.35,1.24,1.20,1.19,1.18,1.18,1.14])
    k=CubicSpline(gref,kref)(gmag)
    geref=np.array([6.00,12.50,13.50,14.50,15.50,16.50,17.50])
    keref=np.array([0.50,0.50,1.01,1.28,1.38,1.44,1.32])
    ke=CubicSpline(geref,keref)(gmag)
    k=np.where(ruwe>1.4,k*(1+ke),k); k=np.where(np.array(npar)==6,k*1.25,k)
    return np.sqrt((spi*k)**2+0.0103**2)

print("Step 8: applying parallax corrections ...", flush=True)
Gmag     = np.array(merged['phot_g_mean_mag'])
nueff    = np.array(merged['nu_eff_used_in_astrometry'])
pscol    = np.array(merged['pseudocolour'])
eclat    = np.array(merged['ecl_lat'])
RUWE     = np.array(merged['ruwe'])
npar_raw = np.array(merged['astrometric_params_solved'])
parrobs  = np.array(merged['parallax'])
eparr    = np.array(merged['parallax_error'])

nparam   = np.where(npar_raw==31, 5, 6)
nueffnew = np.where(npar_raw==31, nueff, pscol)
Gmagc    = np.clip(Gmag, 6.0, None)

ZP = np.array(ZPEDR3(Gmagc, nueffnew, eclat, nparam)) / 1000.
merged['parallax_corrected']       = parrobs - ZP
merged['parallax_error_corrected'] = SPICOR(eparr, Gmagc, RUWE, nparam)
print(f"  ZP mean = {np.nanmean(ZP):.4f} mas"
      f"  |  before: {1000/np.nanmean(parrobs):.0f} pc"
      f"  |  after:  {1000/np.nanmean(parrobs-ZP):.0f} pc")

merged.write(f"{path_data_processed}catalog_complete_{name_complex}.fits", format='fits', overwrite=True)
print(f"Saved -> {path_data_processed}catalog_complete_{name_complex}.fits  ({len(merged)} sources)")
print(f"Parallax range: {np.nanmin(merged['parallax_corrected']):.3f} to {np.nanmax(merged['parallax_corrected']):.3f} mas")
print(f"Distance range: {1000/np.nanmax(merged['parallax_corrected']):.0f} to {1000/np.nanmin(merged['parallax_corrected']):.0f} pc")
print(f"G mag range: {np.nanmin(Gmag):.1f} to {np.nanmax(Gmag):.1f}")
