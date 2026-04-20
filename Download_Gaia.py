#!/usr/bin/env python
"""
Script 1 — Download from Gaia archive and apply parallax correction
===================================================================
Queries Gaia DR3 + BailerJones distances for a RA/Dec rectangular box,
cross-matches 2MASS and PanSTARRS via the DR3 neighbour tables,
applies the Lindegren+2021 zero-point and Maiz Apellaniz 2022
uncertainty correction, then saves a single merged FITS file.

Output : <path_out>/catalog_complete_<name>.fits
"""

import numpy as np
import warnings
from astropy.table import Table, join, unique
from astropy.utils.exceptions import AstropyWarning
from astroquery.gaia import Gaia
from scipy.interpolate import CubicSpline

warnings.simplefilter('ignore', AstropyWarning)
Gaia.ROW_LIMIT = -1

# ----------- Parameters (edit for your field) 
path_data	 = 'data/'
name_complex = 'Orion_OB1'

# Sky box in RA/Dec — edit to match your paper's field definition
RA_MIN  =  75.0   # deg
RA_MAX  =  90.0   # deg
DEC_MIN = -14.0   # deg
DEC_MAX =  16.0   # deg

# Astrometric quality cuts
MAX_DIST_PC = 1000.0   # BailerJones r_med_geo upper cut [pc]
MAX_G       =   19.0   # faint G-mag limit
MIN_PLX     =    0.5   # parallax lower cut [mas]
MIN_PLX_SNR =    5.0   # parallax / error cut

# ==============================================================================
# 1. Gaia DR3 + BailerJones
# ==============================================================================
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
    g.ruwe, g.astrometric_params_solved,
    d.r_med_geo, d.r_lo_geo, d.r_hi_geo
FROM gaiadr3.gaia_source AS g
INNER JOIN external.gaiaedr3_distance AS d ON g.source_id = d.source_id
WHERE g.ra  BETWEEN {RA_MIN}  AND {RA_MAX}
  AND g.dec BETWEEN {DEC_MIN} AND {DEC_MAX}
  AND d.r_med_geo > 0 AND d.r_med_geo < {MAX_DIST_PC}
  AND g.phot_g_mean_mag     < {MAX_G}
  AND g.parallax            > {MIN_PLX}
  AND g.parallax_over_error > {MIN_PLX_SNR}
"""
print("Querying Gaia DR3 + BailerJones ...")
gaia_t = Gaia.launch_job_async(gaia_query).get_results()
print(f"  {len(gaia_t)} sources")
gaia_t.write(path + 'raw_gaia.fits', format='fits', overwrite=True)

# ==============================================================================
# 2. Upload source IDs for server-side join
# ==============================================================================
Gaia.upload_table(
    upload_resource=Table({'source_id': np.array(gaia_t['source_id'], dtype=np.int64)}),
    table_name='my_sources')

# ==============================================================================
# 3. 2MASS (three-table chain: best_neighbour -> join -> original_valid)
# ==============================================================================
tmass_query = """
SELECT bn.source_id AS dr3_source_id,
       tmass.j_m, tmass.j_msigcom,
       tmass.h_m, tmass.h_msigcom,
       tmass.ks_m, tmass.ks_msigcom, tmass.ph_qual
FROM user_upload.my_sources AS up
INNER JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS bn ON up.source_id = bn.source_id
INNER JOIN gaiadr3.tmass_psc_xsc_join AS xj ON bn.clean_tmass_psc_xsc_oid = xj.clean_tmass_psc_xsc_oid
INNER JOIN gaiadr2.tmass_original_valid AS tmass ON xj.original_psc_source_id = tmass.designation
"""
print("Querying 2MASS ...")
tmass_t = Gaia.launch_job_async(tmass_query).get_results()
print(f"  {len(tmass_t)} rows")
tmass_t.write(path + 'raw_2mass.fits', format='fits', overwrite=True)

# ==============================================================================
# 4. PanSTARRS DR1
# ==============================================================================
ps_query = """
SELECT bn.source_id AS dr3_source_id,
       ps.g_mean_psf_mag, ps.g_mean_psf_mag_error,
       ps.r_mean_psf_mag, ps.r_mean_psf_mag_error,
       ps.i_mean_psf_mag, ps.i_mean_psf_mag_error,
       ps.z_mean_psf_mag, ps.z_mean_psf_mag_error,
       ps.y_mean_psf_mag, ps.y_mean_psf_mag_error
FROM user_upload.my_sources AS up
INNER JOIN gaiadr3.panstarrs1_best_neighbour AS bn ON up.source_id = bn.source_id
INNER JOIN external.panstarrs1_original_valid AS ps ON bn.original_ext_source_id = ps.obj_id
"""
print("Querying PanSTARRS DR1 ...")
ps_t = Gaia.launch_job_async(ps_query).get_results()
print(f"  {len(ps_t)} rows")
ps_t.write(path + 'raw_panstarrs.fits', format='fits', overwrite=True)

# ==============================================================================
# 5. Rybizki+2022 fidelity, cross-match via Gaia TAP
#    The fidelity table is hosted on the Gaia archive as an external catalog.
#    We join on source_id using the already-uploaded my_sources table.
# ==============================================================================
fidelity_query = """
SELECT up.source_id, f.fidelity_v2 AS fidelity
FROM user_upload.my_sources AS up
INNER JOIN external.gaiadr3_spurious AS f ON up.source_id = f.source_id
"""
print("Querying Rybizki+2022 fidelity ...")
try:
    fidelity_t = Gaia.launch_job_async(fidelity_query).get_results()
    print(f"  {len(fidelity_t)} rows with fidelity")
    fidelity_t.write(path + 'raw_fidelity.fits', format='fits', overwrite=True)
except Exception as e:
    print(f"  Fidelity query failed ({e})")
    print("  Continuing without fidelity, RUWE-only cut will be used in script 2")
    fidelity_t = None

# ==============================================================================
# 6. Merge
# ==============================================================================
print("Merging ...")
tmass_clean = unique(tmass_t, keys='dr3_source_id')
merged = join(gaia_t,  tmass_clean, keys_left='source_id', keys_right='dr3_source_id', join_type='left')
merged = join(merged,  ps_t,        keys_left='source_id', keys_right='dr3_source_id', join_type='left')
if fidelity_t is not None:
    merged = join(merged, fidelity_t, keys='source_id', join_type='left')
    n_fid = np.sum(np.isfinite(np.array(merged['fidelity'], dtype=float)))
    print(f"  Fidelity matched: {n_fid}")
assert len(merged) == len(gaia_t), f"Row count changed: {len(gaia_t)} -> {len(merged)}"
print(f"  Total: {len(merged)}"
      f"  |  2MASS: {np.sum(np.isfinite(np.array(merged['j_m'], dtype=float)))}"
      f"  |  PS: {np.sum(np.isfinite(np.array(merged['g_mean_psf_mag'], dtype=float)))}")

# ==============================================================================
# 6. Parallax zero-point correction (Lindegren+2021 + Maiz Apellaniz 2022)
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
    return [z5[j] if npar[j]==5 else z6[j] for j in range(n)]   # µas

def SPICOR(spi, gmag, ruwe, npar):
    gref=np.array([6.50,7.50,8.50,9.50,10.25,10.75,11.25,11.75,12.25,12.75,13.25,13.75,14.25,14.75,15.25,15.75,16.25,16.75,17.25,17.75])
    kref=np.array([2.62,2.38,2.06,1.66,1.22,1.41,1.76,1.76,1.90,1.92,1.61,1.50,1.39,1.35,1.24,1.20,1.19,1.18,1.18,1.14])
    k=CubicSpline(gref,kref)(gmag)
    geref=np.array([6.00,12.50,13.50,14.50,15.50,16.50,17.50])
    keref=np.array([0.50,0.50,1.01,1.28,1.38,1.44,1.32])
    ke=CubicSpline(geref,keref)(gmag)
    k=np.where(ruwe>1.4,k*(1+ke),k); k=np.where(np.array(npar)==6,k*1.25,k)
    return np.sqrt((spi*k)**2+0.0103**2)

print("Applying parallax corrections ...")
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

ZP = np.array(ZPEDR3(Gmagc, nueffnew, eclat, nparam)) / 1000.   # µas → mas
merged['parallax_corrected']       = parrobs - ZP
merged['parallax_error_corrected'] = SPICOR(eparr, Gmagc, RUWE, nparam)
print(f"  ZP mean = {np.nanmean(ZP):.4f} mas"
      f"  |  before: {1000/np.nanmean(parrobs):.0f} pc"
      f"  |  after:  {1000/np.nanmean(parrobs-ZP):.0f} pc")

merged.write(f"{path_data}catalog_complete_{name_complex}.fits", format='fits', overwrite=True)
print(f"Saved -> {path_data}catalog_complete_{name_complex}.fits  ({len(merged)} sources)")
print(f"Parallax range: {np.nanmin(merged['parallax_corrected']):.3f} to {np.nanmax(merged['parallax_corrected']):.3f} mas")
print(f"Distance range: {1000/np.nanmax(merged['parallax_corrected']):.0f} to {1000/np.nanmin(merged['parallax_corrected']):.0f} pc")
print(f"G mag range: {np.nanmin(Gmag):.1f} to {np.nanmax(Gmag):.1f}")
