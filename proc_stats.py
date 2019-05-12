from lib.fnc_matching import *
import os
import json
import numpy as np
import multiprocessing as mp
from itertools import product

ffindids = "data/matching_find_ids.json"
fmatching = "data/matching.npy"
fstats = "data/stats.json"

def stat_worker(weights_mp, collect_mp, distance):
	
	while True:
		try:
			w_R, w_th, w_kap, w_h, w_diam, w_ax = weights_mp.pop()
		except:
			break
		print("\rcombs left: %d                       " % (len(weights_mp)), end = "")
		D = combine_dists(distance, w_R, w_th, w_kap, w_h, w_diam, w_ax)
		
		pca = PCA(n_components = None)
		pca.fit(D)
		n_components = np.where(np.cumsum(pca.explained_variance_ratio_) > 0.9)[0].min()
		pca = PCA(n_components = n_components)
		pca.fit(D)
		pca_scores = pca.transform(D)
		
		z = linkage(pca_scores, method = "ward")
		
		iters = 100
		
		n_samples = distance.shape[0]
		n_select = int(round(n_samples / 2))
		sample_idxs = np.arange(n_samples)
		lda = LinearDiscriminantAnalysis()
		ps = []
		max_clusters = min(n_select // 2, n_samples // 8)
		for n_clusters in range(2, max_clusters):
			clusters = np.array(fcluster(z, n_clusters, "maxclust"), dtype = int)
			matches = 0
			for r in range(iters):
				idxs = np.random.choice(sample_idxs, n_select, replace = False)
				inv_idxs = np.setdiff1d(sample_idxs, idxs, assume_unique = True)
				lda.fit(pca_scores[idxs], clusters[idxs])
				pred = lda.predict(pca_scores[inv_idxs])
				matches += (pred == clusters[inv_idxs]).sum()
			n_clusters = clusters.max()
			
			p = matches / ((iters - iters*n_clusters)*n_select + (iters*n_clusters - iters)*n_samples - matches*n_clusters + 2*matches)
			# p = probability that a positive match is not random
			
			ps.append([n_clusters, p])

		ps = np.array(ps)
		
		clu_index = np.sqrt(ps[:,0]*ps[:,1])
		n_clusters_idx, p_opt = ps[np.argmax(clu_index)]
		n_clusters_p = ps[ps[:,1] > 0.5,0].max()
		
		collect_mp.append([w_R, w_th, w_kap, w_h, w_diam, w_ax, n_clusters_idx, n_clusters_p, p_opt, clu_index.max()])

if __name__ == '__main__':
	
	with open(ffindids, "r") as f:
		find_ids = json.load(f)
	
	steps = 20
	vars = 3
	weights = set()
	values = np.round(np.linspace(0, 1, steps + 1)[1:-1], 2).tolist()
	for row in product(*[values for i in range(vars)]):
		if sum(row) != 1:
			continue
		weights.add(tuple([0, 0, 0] + list(row)))
	weights = sorted(weights)
	
	# weights = ((w_R, w_th, w_kap, w_h, w_diam, w_ax), ...)
	
	distance = np.load(fmatching) # distance[i, j] = [R_dist, th_dist, kap_dist, h_dist, diam_dist, ax_dist]
	
	manager = mp.Manager()
	weights_mp = manager.list(weights)
	collect_mp = manager.list()
	
	procs = []
	for pi in range(mp.cpu_count()):
		procs.append(mp.Process(target = stat_worker, args = (weights_mp, collect_mp, distance)))
		procs[-1].start()
	for proc in procs:
		proc.join()
	for proc in procs:
		proc.terminate()
		proc = None
	
	stats = sorted(list(collect_mp))  # [[w_R, w_th, w_kap, w_h, w_diam, w_ax, n_clusters_idx, n_clusters_p, p_opt, max_clu_index], ...]; p_opt = bayesian probability of LDA assigning a profile to the right cluster
	
	with open(fstats, "w") as f:
		json.dump(stats, f)

