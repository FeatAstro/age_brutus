#!/usr/bin/env python
"""
Script 3b — Dynesty isochrone fitting for one HDBSCAN cluster
==============================================================
Fits population parameters using nested sampling.
Run one cluster at a time:

    python script3_fit.py <cluster_id>
    python script3_fit.py <cluster_id> --resume

Output : outputs/brutus_<name_complex>/cluster_<id>_<name_cluster>/samples.npy
                                    				results.npy
                                    				logz.npy
                                    				corner.png
                                    				checkpoint.save
                                    				powell_best.npy
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys

import gc
import time
import numpy as np
import dynesty
from dynesty import utils as dyfunc
from multiprocessing import Pool
from astropy.table import Table
from scipy.optimize import minimize

from brutus.utils import inv_magnitude
from brutus.data import filters as bfilters
from tutorial_utils import find_brutus_data_file

# ----------- Paths 
path_data    = 'data/processed/'
path_out 	 = 'outputs/brutus_'
name_complex = 'Orion_OB1'
name_cluster = 'ONC'
out_base     = path_out + name_complex + '/'
os.makedirs(path_data, exist_ok=True)
os.makedirs(path_out, exist_ok=True)

# ----------- Global parameters 
MIN_SAMPLES  = 37
PMEM_CUT     = 0.5
N_WORKERS    = 6
NLIVE        = 200

# ==============================================================================
# Parameter configuration
# ==============================================================================
# Set fixed=None to sample that parameter freely.
# Set fixed=<value> to hold it constant.
# lo, hi define the prior range (used whether free or fixed for bounds checking).
# init is the Powell starting guess (only used if free).
# below an example for the ONC cluster
#
#               fixed    lo      hi     init
PARAMS = {
    'feh'   : ( 0.0,   -0.5,    0.5,    0.0  ),  # metallicity [Fe/H]
    'loga'  : ( None,   5.5,    7.0,    6.3  ),  # log10(age/yr)
    'av'    : ( None,   0.1,    5.0,    1.0  ),  # V-band extinction
    'rv'    : ( 3.1,    2.0,    5.0,    3.1  ),  # extinction law R(V)
    'dist'  : ( None,   300,    500,  400.0  ),  # distance [pc]
    'ffield': ( 0.05,  1e-6,    0.9,    0.05 ),  # field contamination fraction
}
# ==============================================================================

# internal — derived from PARAMS, do not edit below
_ALL_PARAMS  = ['feh', 'loga', 'av', 'rv', 'dist', 'ffield']
_FREE_PARAMS = [p for p in _ALL_PARAMS if PARAMS[p][0] is None]
_FIXED_VALS  = {p: PARAMS[p][0] for p in _ALL_PARAMS if PARAMS[p][0] is not None}
_NDIM        = len(_FREE_PARAMS)

# filter set
combined_filters = bfilters.gaia + bfilters.ps[:5] + bfilters.tmass

# gaia magnitude floor errors
SIGMA_G  = 0.0027553202
SIGMA_BP = 0.0027901700
SIGMA_RP = 0.0037793818

# ==============================================================================
# Parse arguments
# ==============================================================================
if len(sys.argv) < 2:
    print("Usage: python script3_fit.py <cluster_id> [--resume]")
    sys.exit(1)

CLUSTER_ID  = int(sys.argv[1])
RESUME_MODE = '--resume' in sys.argv

out_dir         = out_base + f'cluster_{CLUSTER_ID}_{name_cluster}/'
CHECKPOINT_FILE = out_dir + 'checkpoint.save'
os.makedirs(out_dir, exist_ok=True)

print(f"Cluster ID    : {CLUSTER_ID}", flush=True)
print(f"Output dir    : {out_dir}", flush=True)
print(f"Resume mode   : {RESUME_MODE}", flush=True)
print(f"Free  ({_NDIM}D) : {_FREE_PARAMS}", flush=True)
print(f"Fixed         : {_FIXED_VALS}", flush=True)

# ==============================================================================
# Photometry builder
# ==============================================================================
def build_photometry(members):
    g_err  = np.sqrt((1.086 / np.array(members['phot_g_mean_flux_over_error'],  dtype=float))**2 + SIGMA_G**2)
    bp_err = np.sqrt((1.086 / np.array(members['phot_bp_mean_flux_over_error'], dtype=float))**2 + SIGMA_BP**2)
    rp_err = np.sqrt((1.086 / np.array(members['phot_rp_mean_flux_over_error'], dtype=float))**2 + SIGMA_RP**2)

    all_mag = np.column_stack([
        np.array(members['G_mag'],  dtype=float),
        np.array(members['BP_mag'], dtype=float),
        np.array(members['RP_mag'], dtype=float),
        np.array(members['g_mean_psf_mag'], dtype=float),
        np.array(members['r_mean_psf_mag'], dtype=float),
        np.array(members['i_mean_psf_mag'], dtype=float),
        np.array(members['z_mean_psf_mag'], dtype=float),
        np.array(members['y_mean_psf_mag'], dtype=float),
        np.array(members['j_m'],  dtype=float),
        np.array(members['h_m'],  dtype=float),
        np.array(members['ks_m'], dtype=float),
    ])
    all_magerr = np.column_stack([
        g_err, bp_err, rp_err,
        np.array(members['g_mean_psf_mag_error'], dtype=float),
        np.array(members['r_mean_psf_mag_error'], dtype=float),
        np.array(members['i_mean_psf_mag_error'], dtype=float),
        np.array(members['z_mean_psf_mag_error'], dtype=float),
        np.array(members['y_mean_psf_mag_error'], dtype=float),
        np.array(members['j_msigcom'],  dtype=float),
        np.array(members['h_msigcom'],  dtype=float),
        np.array(members['ks_msigcom'], dtype=float),
    ])
    phot, err = inv_magnitude(all_mag, all_magerr)
    err  = np.sqrt(err**2 + (0.02 * phot)**2)
    mask = np.isfinite(phot) & (err > 0) & (phot > 0)
    phot = np.where(mask, phot, 1.0)
    err  = np.where(mask, err,  1.0)
    return phot, err, mask


# ==============================================================================
# Module-level globals — populated before forking workers
# ==============================================================================
stellarpop     = None
phot_q         = None
err_q          = None
parallax_q     = None
parallax_err_q = None
pmem_q         = None
mask_q         = None


def load_stellarpop():
    global stellarpop
    from brutus.core import Isochrone, StellarPop
    iso        = Isochrone(mistfile=find_brutus_data_file('MIST_1.2_iso_vvcrit0.0.h5'))
    stellarpop = StellarPop(iso, nnfile=find_brutus_data_file('nn_c3k.h5'),
                            filters=combined_filters)


def prior_transform(u):
    theta = np.zeros(_NDIM)
    for i, p in enumerate(_FREE_PARAMS):
        lo, hi = PARAMS[p][1], PARAMS[p][2]
        theta[i] = lo + u[i] * (hi - lo)
    return theta


def log_likelihood(theta_free):
    # reconstruct full 6-parameter vector from free + fixed
    vals = dict(zip(_FREE_PARAMS, theta_free))
    vals.update(_FIXED_VALS)
    theta_full = np.array([vals[p] for p in _ALL_PARAMS])
    theta_full[5] = np.clip(theta_full[5], 1e-6, 1.0 - 1e-6)
    try:
        from brutus.analysis import isochrone_population_loglike
        lnl = isochrone_population_loglike(
            theta_full, stellarpop, phot_q, err_q,
            parallax=parallax_q, parallax_err=parallax_err_q,
            cluster_prob=pmem_q, mask=mask_q,
        )
        return float(lnl) if np.isfinite(lnl) else -1e30
    except Exception:
        return -1e30


# ==============================================================================
# Main
# ==============================================================================
if __name__ == '__main__':
    import psutil

    catalog = Table.read(path_data + f'hdbscan_clusters_{name_complex}_ms{MIN_SAMPLES}.fits')
    mask_cl = np.array(catalog['cluster_id']) == CLUSTER_ID
    if not np.any(mask_cl):
        print(f"Cluster ID {CLUSTER_ID} not found.")
        sys.exit(1)

    members = catalog[mask_cl]
    pmem    = np.array(members['probability'], dtype=float)

    phot, err, band_mask = build_photometry(members)
    parallax     = np.array(members['parallax_corrected'],       dtype=float)
    parallax_err = np.array(members['parallax_error_corrected'], dtype=float)

    n_valid = np.sum(band_mask, axis=1)
    quality = (pmem > PMEM_CUT) & (n_valid >= 3) & np.isfinite(parallax)
    print(f"Total members     : {len(members)}", flush=True)
    print(f"After quality cut : {quality.sum()}", flush=True)

    if quality.sum() < 5:
        print("Too few members after quality cut — aborting.")
        sys.exit(1)

    # set module globals
    phot_q         = phot[quality]
    err_q          = err[quality]
    parallax_q     = parallax[quality]
    parallax_err_q = parallax_err[quality]
    pmem_q         = np.clip(pmem[quality], 1e-6, 1.0)
    mask_q         = band_mask[quality]

    print(f"Available RAM     : {psutil.virtual_memory().available/1024**3:.1f} GB", flush=True)

    # load stellarpop in main process
    load_stellarpop()

    # sanity check
    print("Sanity check ...", flush=True)
    theta_test = np.array([PARAMS[p][3] for p in _FREE_PARAMS])
    test_val   = log_likelihood(theta_test)
    print(f"  log_likelihood(init) = {test_val:.2f}", flush=True)
    if test_val == -1e30:
        raise RuntimeError("Likelihood returned -1e30 — check data arrays.")
    t0 = time.time()
    log_likelihood(theta_test)
    print(f"  single call: {time.time()-t0:.2f}s", flush=True)
    print("  OK.", flush=True)

    # Powell optimisation
    print("Running Powell ...", flush=True)

    def neg_loglike(theta_free):
        for i, p in enumerate(_FREE_PARAMS):
            lo, hi = PARAMS[p][1], PARAMS[p][2]
            if not (lo < theta_free[i] < hi):
                return 1e30
        lnl = log_likelihood(theta_free)
        return -lnl if np.isfinite(lnl) else 1e30

    theta_init    = np.array([PARAMS[p][3] for p in _FREE_PARAMS])
    powell_bounds = [(PARAMS[p][1], PARAMS[p][2]) for p in _FREE_PARAMS]
    result        = minimize(neg_loglike, theta_init, method='Powell',
                             bounds=powell_bounds, options={'maxiter': 300, 'ftol': 0.1})
    theta_best    = result.x
    print(f"  Powell converged: {result.success}", flush=True)
    for p, v in zip(_FREE_PARAMS, theta_best):
        lo, hi = PARAMS[p][1], PARAMS[p][2]
        flag = '  !! near bound' if abs(v-lo) < 0.05*(hi-lo) or abs(v-hi) < 0.05*(hi-lo) else ''
        print(f"    {p:8s} = {v:.4f}{flag}", flush=True)
    np.save(out_dir + 'powell_best.npy', theta_best)

    # generate live points in main process before releasing stellarpop
    print("Generating live points ...", flush=True)
    rng       = np.random.default_rng(42)
    live_u    = rng.uniform(size=(NLIVE, _NDIM))
    live_v    = np.array([prior_transform(u) for u in live_u])
    live_logl = np.zeros(NLIVE)
    for i, v in enumerate(live_v):
        live_logl[i] = log_likelihood(v)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{NLIVE}", flush=True)
    print(f"  done, min logl: {live_logl.min():.1f}", flush=True)

    # release before forking
    stellarpop = None
    gc.collect()

    # nested sampling
    with Pool(processes=N_WORKERS, initializer=load_stellarpop) as pool:

        if RESUME_MODE and os.path.exists(CHECKPOINT_FILE):
            print(f"Resuming from {CHECKPOINT_FILE}", flush=True)
            sampler = dynesty.NestedSampler.restore(CHECKPOINT_FILE, pool=pool)
        else:
            if RESUME_MODE:
                print("No checkpoint found — starting fresh.", flush=True)
            sampler = dynesty.NestedSampler(
                log_likelihood,
                prior_transform,
                ndim=_NDIM,
                nlive=NLIVE,
                live_points=(live_u, live_v, live_logl),
                pool=pool,
                queue_size=N_WORKERS * 2,
                use_pool={'prior_transform': False, 'loglikelihood': True},
                sample='rwalk',
                bound='single',
                walks=25,
            )

        print("Running nested sampling ...", flush=True)
        sampler.run_nested(
            print_progress=True,
            checkpoint_file=CHECKPOINT_FILE,
            checkpoint_every=300,
        )
        print("Done.", flush=True)

    results = sampler.results

    log_z     = results.logz[-1]
    log_z_err = results.logzerr[-1]
    print(f"log Z = {log_z:.2f} +/- {log_z_err:.2f}", flush=True)

    weights      = np.exp(results.logwt - results.logz[-1])
    flat_samples = dyfunc.resample_equal(results.samples, weights)
    print(f"Posterior samples: {len(flat_samples)}", flush=True)

    q16, q50, q84 = np.percentile(flat_samples, [16, 50, 84], axis=0)
    print("\nResults:", flush=True)
    for i, p in enumerate(_FREE_PARAMS):
        print(f"  {p:8s} = {q50[i]:.4f}  +{q84[i]-q50[i]:.4f} / -{q50[i]-q16[i]:.4f}", flush=True)
    if 'loga' in _FREE_PARAMS:
        idx = _FREE_PARAMS.index('loga')
        print(f"  age      = {10**q50[idx]/1e6:.2f} Myr", flush=True)

    np.save(out_dir + 'samples.npy', flat_samples)
    np.save(out_dir + 'results.npy', results, allow_pickle=True)
    np.save(out_dir + 'logz.npy',    np.array([log_z, log_z_err]))

    # save which parameters were free/fixed for reproducibility
    np.save(out_dir + 'free_params.npy',  np.array(_FREE_PARAMS))
    np.save(out_dir + 'fixed_params.npy', np.array(list(_FIXED_VALS.items()), dtype=object),
            allow_pickle=True)
    print(f"Saved to {out_dir}", flush=True)

    try:
        import corner
        fig = corner.corner(flat_samples, labels=_FREE_PARAMS,
                            quantiles=[0.16, 0.5, 0.84], show_titles=True)
        fig.savefig(out_dir + 'corner.png', dpi=150, bbox_inches='tight')
        print("Corner plot saved.", flush=True)
    except ImportError:
        print("corner not installed, skipping.", flush=True)
