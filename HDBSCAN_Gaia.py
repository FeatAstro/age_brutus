#!/usr/bin/env python
"""
Script 2 — HDBSCAN clustering in 5D phase space
================================================
Same algorithm as Quintana et al. (2025):
  features = (X, Y, Z, cv·Vl, cv·Vb)
  with LSR-corrected proper motions converted to transverse velocities.

Input  : <path_data>/catalog_complete_<name>.fits
Output : <path_out>/hdbscan_clusters_<name>ms<MIN_SAMPLES>.fits
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord, GalacticLSR
import astropy.units as u
from sklearn.cluster import HDBSCAN

path_data    = 'data/processed'
path_out 	 = 'outputs/'
path_im 	 = 'images/HDBSCAN/'
name_complex = 'Orion_OB1'

# ----------- Parameters 
MIN_CLUSTER_SIZE = 15
MIN_SAMPLES      = 37	# change to move to different structure regime 
CV               = 6    # velocity scaling (Kerr+ 2023)

# ==============================================================================
# 1. Load catalog
# ==============================================================================
t = Table.read(path_data + 'catalog_complete_' + name_complex + '.fits')
print(f"Loaded {len(t)} sources")

# ==============================================================================
# 2. Selection cuts (Sanchez-Sanjuan et al. 2024)        
# ==============================================================================

# ----------- Parallax: 3σ cut around TTS median -> 2.0 - 3.6 mas (~277 - 500 pc) 
plx_corr  = np.array(t['parallax_corrected'])
plxe_corr = np.array(t['parallax_error_corrected'])
plx_cut   = (plx_corr >= 2.0) & (plx_corr <= 3.6)

# ----------- Parallax relative uncertainty < 5% 
plx_snr_cut = (plxe_corr / plx_corr) < 0.05

# ----------- Total proper motion: 6*sigma cut < 5.425 mas/yr 
pmra   = np.array(t['pmra'])
pmdec  = np.array(t['pmdec'])
mu     = np.sqrt(pmra**2 + pmdec**2)
mu_cut = mu < 5.425

# ----------- Astrometric quality: RUWE < 1.4 OR fidelity > 0.5 
# Note: fidelity must be cross-matched separately from Rybizki+2022 catalog.
# If fidelity column is not present, only RUWE is used.
ruwe     = np.array(t['ruwe'])
ruwe_cut = ruwe < 1.4
if 'fidelity' in t.colnames:
    fidelity       = np.array(t['fidelity'])
    astrometry_cut = ruwe_cut | (fidelity > 0.5)
    print("  Using RUWE < 1.4 OR fidelity > 0.5")
else:
    astrometry_cut = ruwe_cut
    print("  fidelity column not found, using RUWE < 1.4 only")

# ----------- Colour–magnitude cut: young objects below 30-Myr isochrone 
# M_RP < 2.6 + 2.1*(BP-RP)  for BP-RP > 1.0  (Hernández et al. 2023)
G_mag  = np.array(t['phot_g_mean_mag'])
BP_mag = np.array(t['phot_bp_mean_mag'])
RP_mag = np.array(t['phot_rp_mean_mag'])
dist_pc = 1000.0 / plx_corr                          # simple inversion, only used for CMD cut
M_RP    = RP_mag - 5*np.log10(dist_pc) + 5           # absolute RP magnitude
BP_RP   = BP_mag - RP_mag                             # colour index
cmd_cut = (BP_RP <= 1.0) | (M_RP < 2.6 + 2.1*BP_RP)  # cut only applies for BP-RP > 1.0

# ----------- Combine all cuts 
selection = plx_cut & plx_snr_cut & mu_cut & astrometry_cut & cmd_cut & np.isfinite(plx_corr)

print(f"  parallax 2.0–3.6 mas:     {plx_cut.sum():6d}")
print(f"  parallax SNR > 20 (5%%):  {plx_snr_cut.sum():6d}")
print(f"  proper motion < 5.425:    {mu_cut.sum():6d}")
print(f"  astrometry quality:       {astrometry_cut.sum():6d}")
print(f"  CMD (30-Myr isochrone):   {cmd_cut.sum():6d}")
print(f"  All cuts combined:        {selection.sum():6d} / {len(t)}")

t = t[selection]
print(f"Kept {len(t)} sources after selection")

Glon = np.array(t['l']);    Glat = np.array(t['b'])
ra   = np.array(t['ra']);   dec  = np.array(t['dec'])
ra_error    = np.array(t['ra_error']);    dec_error   = np.array(t['dec_error'])
pmra        = np.array(t['pmra']);        pmdec       = np.array(t['pmdec'])
pmra_error  = np.array(t['pmra_error']); pmdec_error = np.array(t['pmdec_error'])
d   = np.array(t['r_med_geo'])
d16 = np.array(t['r_lo_geo']);  d84 = np.array(t['r_hi_geo'])
ed  = (d84 - d16) / 2.0
e_Plx = (np.array(t['parallax_error_corrected'])
         if 'parallax_error_corrected' in t.colnames
         else np.array(t['parallax_error']))

# ==============================================================================
# 2. Galactic Cartesian positions + errors
# ==============================================================================
c = SkyCoord(l=Glon*u.deg, b=Glat*u.deg, distance=d*u.pc, frame='galactic')
X = c.cartesian.x.value
Y = c.cartesian.y.value
Z = c.cartesian.z.value

def prop_eq_to_gal(ra, dec, ra_e, dec_e):
    theta = ra - np.deg2rad(192.25)
    dl = np.sqrt((np.cos(dec)*np.sin(theta)*ra_e)**2 + (-np.cos(dec)*np.cos(theta)*dec_e)**2)
    db = np.abs(np.cos(dec)) * dec_e
    return dl, db

def prop_gal_to_cart(l, b, dist, l_e, b_e, d_e):
    eX = np.sqrt((-dist*np.cos(b)*np.sin(l)*l_e)**2 + (-dist*np.sin(b)*np.cos(l)*b_e)**2 + (np.cos(b)*np.cos(l)*d_e)**2)
    eY = np.sqrt(( dist*np.cos(b)*np.cos(l)*l_e)**2 + (-dist*np.sin(b)*np.sin(l)*b_e)**2 + (np.cos(b)*np.sin(l)*d_e)**2)
    eZ = np.sqrt((dist*np.cos(b)*b_e)**2 + (np.sin(b)*d_e)**2)
    return eX, eY, eZ

dl, db = prop_eq_to_gal(np.deg2rad(ra), np.deg2rad(dec), ra_error, dec_error)
eX, eY, eZ = prop_gal_to_cart(np.deg2rad(Glon), np.deg2rad(Glat), d, dl, db, ed)

# ==============================================================================
# 3. Proper motions -> Galactic frame
# ==============================================================================
AG = np.matrix([[-0.0548755604162154,-0.8734370902348850,-0.4838350155487132],
                [ 0.4941094278755837,-0.4448296299600112, 0.7469822444972189],
                [-0.8676661490190047,-0.1980763734312015, 0.4559837761750669]])

pml_list, pmb_list, epml_list, epmb_list = [], [], [], []
for j in range(len(ra)):
    ra_r = ra[j]*np.pi/180;   dec_r = dec[j]*np.pi/180
    l_r  = Glon[j]*np.pi/180; b_r   = Glat[j]*np.pi/180

    p_icrs = [-np.sin(ra_r), np.cos(ra_r), 0]
    q_icrs = [-np.cos(ra_r)*np.sin(dec_r), -np.sin(ra_r)*np.sin(dec_r), np.cos(dec_r)]
    p_gal  = [-np.sin(l_r),  np.cos(l_r), 0]
    q_gal  = [-np.cos(l_r)*np.sin(b_r),  -np.sin(l_r)*np.sin(b_r), np.cos(b_r)]

    mu_icrs = np.matrix(p_icrs)*pmra[j] + np.matrix(q_icrs)*pmdec[j]
    mu_gal  = AG * mu_icrs.transpose()
    pml_list.append((np.matrix(p_gal)*mu_gal)[0,0])
    pmb_list.append((np.matrix(q_gal)*mu_gal)[0,0])

    G  = np.matrix([p_gal, q_gal]) * AG * np.matrix([[p_gal[i], q_gal[i]] for i in range(3)])
    J  = np.matrix([[G[0,0],G[0,1],0,0,0],[G[1,0],G[1,1],0,0,0],
                    [0,0,1,0,0],[0,0,0,G[0,0],G[0,1]],[0,0,0,G[1,0],G[1,1]]])
    C  = np.matrix(np.diag([ra_error[j]**2, dec_error[j]**2, e_Plx[j]**2,
                             pmra_error[j]**2, pmdec_error[j]**2]))
    Cg = J * C * J.transpose()
    epml_list.append(np.sqrt(Cg[3,3]))
    epmb_list.append(np.sqrt(Cg[4,4]))

pml = np.array(pml_list);  pmb = np.array(pmb_list)
epml= np.array(epml_list); epmb= np.array(epmb_list)

# ==============================================================================
# 4. LSR correction + transverse velocities
# ==============================================================================
coord = SkyCoord(l=Glon*u.deg, b=Glat*u.deg, distance=d*u.pc,
                 pm_l_cosb=pml*u.mas/u.yr, pm_b=pmb*u.mas/u.yr,
                 radial_velocity=np.zeros(len(d))*u.km/u.s, frame='galactic')
LSR     = coord.transform_to(GalacticLSR)
pmlcorr = LSR.pm_l_cosb.value
pmbcorr = LSR.pm_b.value

Vl  = 4.74 * pmlcorr * d / 1000.
Vb  = 4.74 * pmbcorr * d / 1000.
eVl = np.sqrt((4.74*d/1000.*epml)**2 + (4.74*pmlcorr/1000.*ed)**2)
eVb = np.sqrt((4.74*d/1000.*epmb)**2 + (4.74*pmbcorr/1000.*ed)**2)

# ==============================================================================
# 5. HDBSCAN
# ==============================================================================
Coord = np.vstack((X, Y, Z, CV*Vl, CV*Vb)).T

print(f"Running HDBSCAN (min_cluster_size={MIN_CLUSTER_SIZE}, min_samples={MIN_SAMPLES}) …")
t0 = time.time()
db1 = HDBSCAN(metric='euclidean', min_cluster_size=MIN_CLUSTER_SIZE,
              min_samples=MIN_SAMPLES, cluster_selection_method='leaf')
db1.fit(Coord)
labels        = db1.labels_
probabilities = db1.probabilities_
n_clusters    = len(set(labels)) - (1 if -1 in labels else 0)
print(f"  {time.time()-t0:.1f} s  |  {n_clusters} groups  |  {list(labels).count(-1)} noise")

# ==============================================================================
# 6. Build cluster catalog (same structure as your existing code)
# ==============================================================================
unique_labels    = sorted(set(labels))
core_samples_mask = np.zeros_like(labels, dtype=bool)
cluster_catalog  = []
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0, 0, 0, 1]
        class_member_mask = (labels == k) & ~core_samples_mask
        plt.scatter(Coord[class_member_mask, 0], Coord[class_member_mask, 1],
                    c='0.7', s=5, alpha=0.3)
        continue

    class_member_mask = (labels == k) & ~core_samples_mask
    n = np.sum(class_member_mask)
    plt.scatter(Coord[class_member_mask, 0], Coord[class_member_mask, 1],
                c=[col], s=20, alpha=0.5)

    row = {
        'cluster_id' : np.full(n, k),
        'source_id'  : np.array(t['source_id'])[class_member_mask],
        'ra': ra[class_member_mask],   'dec': dec[class_member_mask],
        'l' : Glon[class_member_mask], 'b'  : Glat[class_member_mask],
        'X' : X[class_member_mask],  'Y': Y[class_member_mask],  'Z': Z[class_member_mask],
        'distance' : d[class_member_mask],
        'Vl': Vl[class_member_mask],  'Vb': Vb[class_member_mask],
        'pml_corr' : pmlcorr[class_member_mask],
        'pmb_corr' : pmbcorr[class_member_mask],
        'X_err': eX[class_member_mask], 'Y_err': eY[class_member_mask], 'Z_err': eZ[class_member_mask],
        'Vl_err': eVl[class_member_mask], 'Vb_err': eVb[class_member_mask],
        'probability': probabilities[class_member_mask],
        'G_mag' : np.array(t['phot_g_mean_mag'])[class_member_mask],
        'BP_mag': np.array(t['phot_bp_mean_mag'])[class_member_mask],
        'RP_mag': np.array(t['phot_rp_mean_mag'])[class_member_mask],
        'phot_g_mean_flux_over_error' : np.array(t['phot_g_mean_flux_over_error'])[class_member_mask],
        'phot_bp_mean_flux_over_error': np.array(t['phot_bp_mean_flux_over_error'])[class_member_mask],
        'phot_rp_mean_flux_over_error': np.array(t['phot_rp_mean_flux_over_error'])[class_member_mask],
        'parallax_corrected'      : np.array(t['parallax_corrected'])[class_member_mask],
        'parallax_error_corrected': np.array(t['parallax_error_corrected'])[class_member_mask],
    }
    for col_name in ['j_m','j_msigcom','h_m','h_msigcom','ks_m','ks_msigcom',
                     'g_mean_psf_mag','g_mean_psf_mag_error',
                     'r_mean_psf_mag','r_mean_psf_mag_error',
                     'i_mean_psf_mag','i_mean_psf_mag_error',
                     'z_mean_psf_mag','z_mean_psf_mag_error',
                     'y_mean_psf_mag','y_mean_psf_mag_error']:
        if col_name in t.colnames:
            row[col_name] = np.array(t[col_name])[class_member_mask]

    cluster_catalog.append(Table(row))

plt.xlabel('X (pc)'); plt.ylabel('Y (pc)')
plt.title(f'Found {n_clusters} stellar groups')
plt.tight_layout(); plt.savefig(path_im + f'hdbscan_{name_complex}ms{MIN_SAMPLES}_XY.png', dpi=150); plt.show()

full_catalog = vstack(cluster_catalog)
out = path_out + f'hdbscan_clusters_{name_complex}ms{MIN_SAMPLES}.fits'
full_catalog.write(out, format='fits', overwrite=True)
print(f"Saved -> {out}  ({len(full_catalog)} members in {len(cluster_catalog)} clusters)")
