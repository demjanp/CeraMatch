from fnc_cluster import *
import numpy as np

import os
import json
from matplotlib import pyplot
from scipy import stats

fmatching = "data/matching.npy"
ffindids = "data/matching_find_ids.json"
fclusters = "data/clusters_manual.json"

if __name__ == '__main__':
	
	D = np.load(fmatching) # D[i, j] = [diam_dist, thick_dist, ax_dist, h_dist, ht_dist]
	
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
	sample_ids = collect
	
	# D[i, j] = [diam_dist, thick_dist, ax_dist, h_dist, ht_dist]
	
	diams_clusters = []
	thick_clusters = []
	axs_clusters = []
	hs_clusters = []
	hts_clusters = []
	
	
	sizes = []
	
	for label in clusters_manual:
		
		samples = [sample_ids.index(sample_id) for sample_id in clusters_manual[label]]
		D_clu = D[:,samples][samples]
		
		diams_clusters += D_clu[:,:,0][D_clu[:,:,0] > 0].flatten().tolist()
		thick_clusters += D_clu[:,:,1][D_clu[:,:,1] > 0].flatten().tolist()
		axs_clusters += D_clu[:,:,2][D_clu[:,:,2] > 0].flatten().tolist()
		hs_clusters += D_clu[:,:,3][D_clu[:,:,3] > 0].flatten().tolist()
		hts_clusters += D_clu[:,:,4][D_clu[:,:,4] > 0].flatten().tolist()
		
	
	diams_all = D[:,:,0][D[:,:,0] > 0].flatten()
	thick_all = D[:,:,1][D[:,:,1] > 0].flatten()
	axs_all = D[:,:,2][D[:,:,2] > 0].flatten()
	hs_all = D[:,:,3][D[:,:,3] > 0].flatten()
	hts_all = D[:,:,4][D[:,:,4] > 0].flatten()
	
	
	perc = 19
	
	print("samples: %d" % (D.shape[0]))
	print("manual clusters: %d" % (len(clusters_manual)))
	
	pyplot.subplot(151)
	pyplot.title("Diameter")
	pyplot.boxplot([diams_all, diams_clusters], sym = ".")
	pyplot.plot([1, 2], [np.percentile(diams_all, perc), np.percentile(diams_clusters, perc)], "+", color = "red")
	pyplot.xticks([1,2], ["All", "Clusters"])
	
	print("diam_lim %d perc.: %f" % (perc, np.percentile(diams_clusters, perc)))
	
	pyplot.subplot(152)
	pyplot.title("Thickness")
	pyplot.boxplot([thick_all, thick_clusters], sym = ".")
	pyplot.plot([1, 2], [np.percentile(thick_all, perc), np.percentile(thick_clusters, perc)], "+", color = "red")
	pyplot.xticks([1,2], ["All", "Clusters"])
	
	print("thick_lim %d perc.: %f" % (perc, np.percentile(thick_clusters, perc)))
	
	pyplot.subplot(153)
	pyplot.title("Axis")
	pyplot.boxplot([axs_all, axs_clusters], sym = ".")
	pyplot.plot([1, 2], [np.percentile(axs_all, perc), np.percentile(axs_clusters, perc)], "+", color = "red")
	pyplot.xticks([1,2], ["All", "Clusters"])
	
	print("ax_lim %d perc.: %f" % (perc, np.percentile(axs_clusters, perc)))
	
	pyplot.subplot(154)
	pyplot.title("Hamming")
	pyplot.boxplot([hs_all, hs_clusters], sym = ".")
	pyplot.plot([1, 2], [np.percentile(hs_all, perc), np.percentile(hs_clusters, perc)], "+", color = "red")
	pyplot.xticks([1,2], ["All", "Clusters"])
	
	print("h_lim %d perc.: %f" % (perc, np.percentile(hs_clusters, perc)))
	
	pyplot.subplot(155)
	pyplot.title("Hamming - Trimmed")
	pyplot.boxplot([hts_all, hts_clusters], sym = ".")
	pyplot.plot([1, 2], [np.percentile(hts_all, perc), np.percentile(hts_clusters, perc)], "+", color = "red")
	pyplot.xticks([1,2], ["All", "Clusters"])
	
	print("ht_lim %d perc.: %f" % (perc, np.percentile(hts_clusters, perc)))
	
	pyplot.tight_layout()
	pyplot.show()
