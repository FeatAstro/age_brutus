#!/usr/bin/env python
"""
Script 3a — Visualise HDBSCAN clusters on a RA/Dec sky map
===========================================================
Optionally cross-matches with a Sanchez-Sanjuan reference catalog
to label clusters with known region names.

Input  : <path_data>/hdbscan_clusters_ms<MIN_SAMPLES>.fits
         <path_data>/Big_Structures_5Dparams_SanchezSanjuan2024_filtered.fits  (optional)
Output : <path_image>/hdbscan_map_ms<MIN_SAMPLES>.png
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, join
import os

path_data    = 'data/'
path_im 	 = 'images/compare/'
os.makedirs(path_image, exist_ok=True)

MIN_SAMPLES = 37

# Sanchez-Sanjuan region names — edit for other complexes or set to None to skip
SANCHEZ_FILE = path_data + 'Big_Structures_5Dparams_SanchezSanjuan2024_filtered.fits'
CLUSTER_NAMES = {
    1: 'lambda Ori', 2: 'Ori-North',  3: 'Briceno-1A', 4: 'Briceno-1B',
    5: 'Ori-East',   6: 'OBP-Far',    7: 'sigma Ori',  8: 'OBP-b',
    9: 'OBP-d',     10: 'OBP-Near',  11: 'ONC',       12: 'Ori-South',
   13: 'Orion Y',
}
REGION_COLORS = {
    'lambda Ori': '#1f77b4', 'Ori-North': '#ff7f0e', 'Briceno-1A': '#2ca02c',
    'Briceno-1B': '#d62728', 'Ori-East':  '#9467bd', 'OBP-Far':    '#8c564b',
    'sigma Ori':  '#e377c2', 'OBP-b':     '#7f7f7f', 'OBP-d':      '#bcbd22',
    'OBP-Near':   '#17becf', 'ONC':       '#ff9896', 'Ori-South':  '#aec7e8',
    'Orion Y':    '#ffbb78',
}
NEW_COLORS = [
    '#e41a1c','#377eb8','#4daf4a','#ff7f00','#a65628',
    '#f781bf','#999999','#984ea3','#ffff33','#66c2a5',
]

# load HDBSCAN catalog
my_clusters = Table.read(path_data + f'hdbscan_clusters_ms{MIN_SAMPLES}.fits')
all_ids     = sorted(set(np.array(my_clusters['cluster_id'])))
print(f"Loaded {len(my_clusters)} members in {len(all_ids)} clusters")

# try to load Sanchez catalog for cross-match 
use_sanchez = os.path.exists(SANCHEZ_FILE)
if use_sanchez:
    known_catalog = Table.read(SANCHEZ_FILE)
    print(f"Sanchez catalog: {len(known_catalog)} stars")
    matched       = join(my_clusters, known_catalog, keys='source_id', join_type='inner')
    sanchez_set   = set(known_catalog['source_id'])
    hdbscan_set   = set(my_clusters['source_id'])
    matched_set   = sanchez_set & hdbscan_set
    new_set       = hdbscan_set - sanchez_set
    recovery      = len(matched_set) / len(known_catalog) * 100
    new_stars     = my_clusters[np.isin(my_clusters['source_id'], list(new_set))]
    print(f"Recovery: {recovery:.1f}%  |  New stars: {len(new_set)}")
else:
    print("Sanchez catalog not found — showing cluster IDs only")
    recovery  = None
    new_stars = None

# assign labels and colors to each HDBSCAN cluster
hdbscan_labels = {}
plot_colors    = dict(REGION_COLORS)
new_idx        = 0

for hdb_id in all_ids:
    if use_sanchez:
        matched_in_cluster = matched[matched['cluster_id'] == hdb_id]
        if len(matched_in_cluster) > 0:
            sanchez_ids, counts = np.unique(matched_in_cluster['Cluster'], return_counts=True)
            best_match = sanchez_ids[np.argmax(counts)]
            name = CLUSTER_NAMES.get(best_match, f'Region_{best_match}')
        else:
            name = f'NEW_{hdb_id}'
            plot_colors[name] = NEW_COLORS[new_idx % len(NEW_COLORS)]
            new_idx += 1
    else:
        name = f'Cluster_{hdb_id}'
        plot_colors[name] = NEW_COLORS[new_idx % len(NEW_COLORS)]
        new_idx += 1
    hdbscan_labels[hdb_id] = name

print("\nCluster assignments:")
for hdb_id in all_ids:
    n = np.sum(np.array(my_clusters['cluster_id']) == hdb_id)
    print(f"  ID {hdb_id:3d} -> {hdbscan_labels[hdb_id]}  ({n} stars)")

# axis limits
ra_pad  = (my_clusters['ra'].max()  - my_clusters['ra'].min())  * 0.05
dec_pad = (my_clusters['dec'].max() - my_clusters['dec'].min()) * 0.05
ra_lim  = (my_clusters['ra'].max()  + ra_pad,  my_clusters['ra'].min()  - ra_pad)
dec_lim = (my_clusters['dec'].min() - dec_pad, my_clusters['dec'].max() + dec_pad)

if use_sanchez:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # left panel: Sanchez catalog
    for cid in np.unique(known_catalog['Cluster']):
        mask = known_catalog['Cluster'] == cid
        name = CLUSTER_NAMES.get(cid, f'Region_{cid}')
        ax1.scatter(known_catalog['ra_1'][mask], known_catalog['dec_1'][mask],
                    s=10, alpha=0.6, color=REGION_COLORS.get(name, 'grey'), label=name)
    ax1.set_title('Sanchez-Sanjuan 2024 Catalog', fontsize=14)
    ax1.legend(fontsize=8)
    ax1.set_xlim(*ra_lim)
    ax1.set_ylim(*dec_lim)
    #ax1.invert_xaxis()
    ax1.set_xlabel('RA (deg)'); ax1.set_ylabel('Dec (deg)')
    ax1.grid(alpha=0.3)

    ax = ax2
    title = (f'HDBSCAN Clusters (labeled with Sanchez-Sanjuan regions)\n'
             f'min_samples = {MIN_SAMPLES}  —  recovery: {recovery:.1f}%')
else:
    fig, ax = plt.subplots(figsize=(9, 7))
    title = f'HDBSCAN Clusters  |  min_samples = {MIN_SAMPLES}'

# right panel (or only panel): HDBSCAN clusters
for hdb_id in all_ids:
    mask  = np.array(my_clusters['cluster_id']) == hdb_id
    label = hdbscan_labels[hdb_id]
    color = plot_colors.get(label, 'grey')
    ax.scatter(my_clusters['ra'][mask], my_clusters['dec'][mask],
               s=15, alpha=0.6, color=color,
               label=f'{hdb_id}: {label}')

if new_stars is not None and len(new_stars) > 0:
    ax.scatter(new_stars['ra'], new_stars['dec'],
               s=10, alpha=0.5, marker='o', facecolors='none',
               edgecolors='black', linewidths=0.8,
               label=f'New stars ({len(new_stars)})', zorder=10)

ax.set_title(title, fontsize=12)
ax.legend(fontsize=8)
ax.set_xlim(*ra_lim)
ax.set_ylim(*dec_lim)
#ax.invert_xaxis()
ax.set_xlabel('RA (deg)'); ax.set_ylabel('Dec (deg)')
ax.grid(alpha=0.3)

plt.tight_layout()
out = path_image + f'hdbscan_map_ms{MIN_SAMPLES}.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {out}")
