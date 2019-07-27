from fnc_cluster import *
import numpy as np

import os
import json
from matplotlib import pyplot
from scipy import stats

fmatching = "data/matching.npy"
fmatching_alt = "data/matching_alt.npy"
ffindids = "data/matching_find_ids.json"

fclusters = "data/clusters_manual.json"

if __name__ == '__main__':
	
	D = np.load(fmatching) # D[i, j] = [R_dist, th_dist, kap_dist, unwrap_dist, diam_dist, ax_dist]
	D_alt = np.load(fmatching_alt)
	
	with open(ffindids, "r") as f:
		sample_ids = json.load(f)

	with open(fclusters, "r") as f:
		clusters_manual = json.load(f)  # {label: [sample_id, ...], ...}
	collect = {}
	for label in clusters_manual:
		if label.startswith("1."):
			continue
		collect[label] = clusters_manual[label]
	clusters_manual = collect
	
	collect = []
	for sample_id in sample_ids:
		has_manual_cluster = False
		for label in clusters_manual:
			if sample_id in clusters_manual[label]:
				has_manual_cluster = True
				break
		if not has_manual_cluster:
			continue
		collect.append(sample_id)
	samples = [sample_ids.index(sample_id) for sample_id in collect]
	D = D[:,samples][samples]
	D_alt = D_alt[:,samples][samples]
	sample_ids = collect
	
	# D[i, j] = [diam_dist, ax_dist, h_dist]
	# D_alt[i, j] = [R_dist, th_dist, kap_dist]
	
	Rs_clusters = []
	ths_clusters = []
	kaps_clusters = []
	diams_clusters = []
	axs_clusters = []
	hs_clusters = []
	
	sizes = []
	
	for label in clusters_manual:
		
		samples = [sample_ids.index(sample_id) for sample_id in clusters_manual[label]]
		D_clu = D[:,samples][samples]
		D_alt_clu = D_alt[:,samples][samples]
		
		Rs_clusters += D_alt_clu[:,:,0][D_alt_clu[:,:,0] > 0].flatten().tolist()
		ths_clusters += D_alt_clu[:,:,1][D_alt_clu[:,:,1] > 0].flatten().tolist()
		kaps_clusters += D_alt_clu[:,:,2][D_alt_clu[:,:,2] > 0].flatten().tolist()
		
		diams_clusters += D_clu[:,:,0][D_clu[:,:,0] > 0].flatten().tolist()
		axs_clusters += D_clu[:,:,1][D_clu[:,:,1] > 0].flatten().tolist()
		hs_clusters += D_clu[:,:,2][D_clu[:,:,2] > 0].flatten().tolist()
	
	Rs_all = D_alt[:,:,0][D_alt[:,:,0] > 0].flatten().tolist()
	ths_all = D_alt[:,:,1][D_alt[:,:,1] > 0].flatten().tolist()
	kaps_all = D_alt[:,:,2][D_alt[:,:,2] > 0].flatten().tolist()
	
	diams_all = D[:,:,0][D[:,:,0] > 0].flatten().tolist()
	axs_all = D[:,:,1][D[:,:,1] > 0].flatten().tolist()
	hs_all = D[:,:,2][D[:,:,2] > 0].flatten().tolist()
	
	pyplot.subplot(231)
	pyplot.title("Radius")
	pyplot.boxplot([Rs_all, Rs_clusters], sym = "")
	pyplot.xticks([1,2], ["All", "Clusters"])
	
	pyplot.subplot(232)
	pyplot.title("Tangent")
	pyplot.boxplot([ths_all, ths_clusters], sym = "")
	pyplot.xticks([1,2], ["All", "Clusters"])
	
	pyplot.subplot(233)
	pyplot.title("Curvature")
	pyplot.boxplot([kaps_all, kaps_clusters], sym = "")
	pyplot.xticks([1,2], ["All", "Clusters"])
	
	pyplot.subplot(234)
	pyplot.title("Diameter")
	pyplot.boxplot([diams_all, diams_clusters], sym = "")
	pyplot.xticks([1,2], ["All", "Clusters"])
	
	pyplot.subplot(235)
	pyplot.title("Axis")
	pyplot.boxplot([axs_all, axs_clusters], sym = "")
	pyplot.xticks([1,2], ["All", "Clusters"])
	
	pyplot.subplot(236)
	pyplot.title("Hamming")
	pyplot.boxplot([hs_all, hs_clusters], sym = "")
	pyplot.xticks([1,2], ["All", "Clusters"])
	
	pyplot.tight_layout()
	pyplot.show()
