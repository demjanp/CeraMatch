import numpy as np
from skimage import measure
import multiprocessing as mp
from natsort import natsorted
from PIL import Image, ImageDraw
from itertools import combinations
from collections import defaultdict
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def reduce_profile(profile, d_min):
	
	return measure.approximate_polygon(np.array(profile), d_min)

def profile_length(profile):
	
	return np.sqrt((np.diff(profile, axis = 0) ** 2).sum(axis = 1)).sum()

def get_rim(profile):
	
	mid_x = profile[:,0].mean()
	left = profile[profile[:,0] <= mid_x]
	right = profile[profile[:,0] >= mid_x]
	rim_left = left[left[:,1] == left[:,1].min()].max(axis = 0)
	rim_right = right[right[:,1] == right[:,1].min()].min(axis = 0)
	if (rim_left[0] == mid_x) and (rim_right[0] == mid_x):
		rim = rim_right
	elif rim_right[0] == mid_x:
		rim = rim_left
	else:
		rim = rim_right
	if rim_left[1] < rim[1]:
		return rim_left
	elif rim_right[1] < rim[1]:
		return rim_right
	return rim

def set_idx0_to_point(coords, point):
	
	idx = np.argmin(cdist([point], coords)[0])
	return np.roll(coords, -idx, axis = 0)

def axis_params(points):
	
	x0, y0 = points[0]
	x1 = points[-1,0]
	y1 = np.poly1d(np.polyfit(points[:,0], points[:,1], 1))(x1)
	xd, yd = x1 - x0, y1 - y0
	div = max(abs(xd), abs(yd))
	if div > 0:
		xd /= div
		yd /= div
	return x0, y0, xd, yd

def diameter_dist(radius1, radius2):
	
	r1, r2 = sorted([radius1, radius2])
	return 1 - r1 / r2
	
def hamming_dist(prof1, prof2, rasterize_factor = 10):
	# return shape matching (morphometric) distance of prof1 and prof2
	# prof1, prof2: [[x,y], ...]
	# prof1 & 2 are assumed to be shifted to 0 ([0,0] coordinate is at the rim)
	# both profiles are trimmed to the length of the shorter
	'''
		M = 1 - (2*Ao) / (A1 + A2)
		M = morphometric distance <0..1>
		Ao = area of overlaping of profiles
		A1, A2 = areas of profiles
	'''
	
	# squared euclidean distances of profile points from 0
	sumsq_1 = (prof1**2).sum(axis = 1)
	sumsq_2 = (prof2**2).sum(axis = 1)
	
	# get maximum distance of a point from 0 from the shorter profile
	maxsq = min(sumsq_1.max(), sumsq_2.max())
	# trim both profiles to the lenght of the shorter
	prof1 = prof1[sumsq_1 <= maxsq]
	prof2 = prof2[sumsq_2 <= maxsq]
	if (prof1.shape[0] <= 1) or (prof2.shape[0] <= 1):
		return 1
	[x0, y0], [x1, y1] = prof1.min(axis = 0), prof1.max(axis = 0)
	[x0_, y0_], [x1_, y1_] = prof2.min(axis = 0), prof2.max(axis = 0)
	x0, y0, x1, y1 = int(min(x0, x0_)) - 1, int(min(y0, y0_)) - 1, int(max(x1, x1_)) + 1, int(max(y1, y1_)) + 1
	w, h = x1 - x0, y1 - y0
	img = Image.new("1", (w * rasterize_factor, h * rasterize_factor))
	draw = ImageDraw.Draw(img)
	prof1 = ((prof1 - [x0, y0]) * rasterize_factor).astype(int)
	draw.polygon([(x, y) for x, y in prof1], fill = 1)
	mask1 = np.array(img, dtype = bool).T
	img.close()
	img = Image.new("1", (w * rasterize_factor, h * rasterize_factor))
	draw = ImageDraw.Draw(img)
	prof2 = ((prof2 - [x0, y0]) * rasterize_factor).astype(int)
	draw.polygon([(x, y) for x, y in prof2], fill = 1)
	mask2 = np.array(img, dtype = bool).T
	img.close()
	
	return 1 - (2*(mask1 & mask2).sum()) / (mask1.sum() + mask2.sum())

def axis_dist(prof1, params1, prof2, params2):
	
	def extract_axis(profile):
		
		idx_rim = np.argmin(cdist([get_rim(profile)], profile)[0])
		axis1, axis2 = profile[:idx_rim][::-1], profile[idx_rim:]
		axis_trim = profile_length(profile) * 0.1
		r_axis1 = profile_length(axis1) / axis_trim
		r_axis2 = profile_length(axis2) / axis_trim
		x2, y2, xd2, yd2 = None, None, None, None
		if (r_axis1 > 0.5) and (r_axis2 > 0.5):
			x2, y2, xd2, yd2 = np.vstack((axis_params(axis1), axis_params(axis2))).T.mean(axis = 1)
		elif r_axis1 > 0.5:
			x2, y2, xd2, yd2 = axis_params(axis1)
		elif r_axis2 > 0.5:
			x2, y2, xd2, yd2 = axis_params(axis2)
		return x2, y2, xd2, yd2
	
	xd2, yd2 = None, None
	sumsq_1 = (prof1**2).sum(axis = 1)
	sumsq_2 = (prof2**2).sum(axis = 1)
	sumsq_1_max, sumsq_2_max = sumsq_1.max(), sumsq_2.max()
	if sumsq_1_max < sumsq_2_max:
		if not params1:
			x1, y1, xd1, yd1 = extract_axis(prof1)
		else:
			x1, y1, xd1, yd1 = params1
		if not params2:
			x2, y2, xd2, yd2 = extract_axis(prof2)
		else:
			x2, y2, xd2, yd2 = params2
		idxs = np.where(sumsq_2 <= sumsq_1_max)[0]
		idx1, idx2 = idxs.min(), idxs.max()
		axis1, axis2 = prof2[:idx1], prof2[idx2:]
	else:
		if not params2:
			x1, y1, xd1, yd1 = extract_axis(prof2)
		else:
			x1, y1, xd1, yd1 = params2
		if not params1:
			x2, y2, xd2, yd2 = extract_axis(prof1)
		else:
			x2, y2, xd2, yd2 = params1
		mask = (sumsq_1 > sumsq_2_max)
		idxs = np.where(sumsq_1 <= sumsq_2_max)[0]
		idx1, idx2 = idxs.min(), idxs.max()
		axis1, axis2 = prof1[:idx1], prof1[idx2:]
	
	axis_trim = max(profile_length(prof1), profile_length(prof2)) * 0.1
	
	r_axis1, r_axis2 = 0, 0
	if (axis1.shape[0] > 1):
		axis1 = axis1[::-1]
		axis1 = axis1[np.hstack(([0], np.sqrt((np.diff(axis1, axis = 0) ** 2).sum(axis = 1)).cumsum())) <= axis_trim]
		r_axis1 = profile_length(axis1) / axis_trim
	if (axis2.shape[0] > 1):
		axis2 = axis2[np.hstack(([0], np.sqrt((np.diff(axis2, axis = 0) ** 2).sum(axis = 1)).cumsum())) <= axis_trim]
		r_axis2 = profile_length(axis2) / axis_trim
	if (r_axis1 > 0.5) and (r_axis2 > 0.5):
		x2, y2, xd2, yd2 = np.vstack((axis_params(axis1), axis_params(axis2))).T.mean(axis = 1)
	elif r_axis1 > 0.5:
		x2, y2, xd2, yd2 = axis_params(axis1)
	elif r_axis2 > 0.5:
		x2, y2, xd2, yd2 = axis_params(axis2)
	
	if [xd2, yd2] == [None, None]:
		return -1
	
	dax = ((yd1 - yd2)**2 + (xd1 - xd2)**2)**0.5 / 8**0.5
	dsq = min(1, (((y1 - y2)**2 + (x1 - x2)**2)**0.5) / (2*(min(sumsq_1_max, sumsq_2_max)**0.5)))
	return ((dax**2 + dsq**2)**0.5) / (2**0.5)

def arc_length(profile):
	
	ds = np.sqrt((np.diff(profile, axis = 0) ** 2).sum(axis = 1))
	s = np.cumsum(ds)
	s -= s[np.argmin(cdist([get_rim(profile)], profile)[0])]
	return s

def tangent(profile):
	
	dx, dy = np.diff(profile, axis = 0).T	
	th = (dy / dx)
	th[dx == 0] = 0
	return th

def fftsmooth(signal):
	
	fft = np.fft.fft(signal)
	freq = np.fft.fftfreq(signal.shape[0])
	cutoff = 0.2
	fft[np.abs(freq) > cutoff] = 0
	filtered = np.fft.ifft(fft)
	return filtered.real

def interpolate_signal(s, signal, s_min, s_max, resolution = 0.1):
	
	xs = np.arange(s_min, s_max + resolution, resolution)
	return xs, np.interp(xs, s, signal)

def radius_dist(prof1, prof2):
	
	R1 = prof1[1:,0]
	R2 = prof2[1:,0]
	s1 = arc_length(prof1)
	s2 = arc_length(prof2)
	s_min, s_max = max(s1.min(), s2.min()), min(s1.max(), s2.max())
	mask1 = (s1 >= s_min) & (s1 <= s_max)
	mask2 = (s2 >= s_min) & (s2 <= s_max)
	
	s1 = s1[mask1]
	xs1, R1 = interpolate_signal(s1, R1[mask1], s_min, s_max)
	s2 = s2[mask2]
	xs2, R2 = interpolate_signal(s2, R2[mask2], s_min, s_max)
	R_dist = np.sqrt(((R1 - R2)**2).sum() / (s_max - s_min))
	
	return R_dist

def dist_worker(ijs_mp, collect_mp, profiles, sample_ids, arc_lengths, tangents, curvatures):
	
	profiles_n = len(profiles)
	distance = np.zeros((profiles_n, profiles_n, 6), dtype = float) - 2
	cnt_last = 0
	cnt = 0
	while True:
		try:
			i, j = ijs_mp.pop()
		except:
			break
		cnt += 1
		if cnt - cnt_last > 100:
			print("\rcombs left: %d                       " % (len(ijs_mp)), end = "")
			cnt_last = cnt
		
		profile1, radius1, params1 = profiles[sample_ids[i]]
		profile2, radius2, params2 = profiles[sample_ids[j]]
		
		shifts1 = profile1[(profile1[:,1] == profile1[:,1].min())][:,0]
		shifts2 = profile2[(profile2[:,1] == profile2[:,1].min())][:,0]
		
		h_dist = np.inf
		ax_dist = np.inf
		R_dist = np.inf
		
		for shift1 in shifts1:
			for shift2 in shifts2:
				h_dist = min(h_dist, hamming_dist(profile1 - [shift1, 0], profile2 - [shift2, 0]))
				ax = axis_dist(profile1 - [shift1, 0], params1, profile2 - [shift2, 0], params2)
				if ax > -1:
					ax_dist = min(ax_dist, ax)
				R_dist = min(R_dist, radius_dist(profile1 - [shift1, 0] + [radius1, 0], profile2 - [shift2, 0] + [radius2, 0]))
		
#		h_dist = hamming_dist(profile1, profile2)
#		ax_dist = axis_dist(profile1, params1, profile2, params2)
#		R_dist = radius_dist(profile1 + [radius1, 0], profile2 + [radius2, 0])
		
		diam_dist = diameter_dist(radius1, radius2)
		
		# shape distances
		s1 = arc_lengths[sample_ids[i]].copy()
		s2 = arc_lengths[sample_ids[j]].copy()
		th1 = tangents[sample_ids[i]].copy()
		th2 = tangents[sample_ids[j]].copy()
		kap1 = curvatures[sample_ids[i]].copy()
		kap2 = curvatures[sample_ids[j]].copy()
		
		s_min, s_max = max(s1.min(), s2.min()), min(s1.max(), s2.max())
		mask1 = (s1 >= s_min) & (s1 <= s_max)
		mask2 = (s2 >= s_min) & (s2 <= s_max)
		
		s1 = s1[mask1]
		_, th1 = interpolate_signal(s1, th1[mask1], s_min, s_max)
		_, kap1 = interpolate_signal(s1, kap1[mask1], s_min, s_max)
		
		s2 = s2[mask2]
		_, th2 = interpolate_signal(s2, th2[mask2], s_min, s_max)
		_, kap2 = interpolate_signal(s2, kap2[mask2], s_min, s_max)
		
		th_dist = np.sqrt(((th1 - th2)**2).sum() / (s_max - s_min))
		kap_dist = np.sqrt(((kap1 - kap2)**2).sum() / (s_max - s_min))

		distance[i, j, 0] = R_dist
		distance[i, j, 1] = th_dist
		distance[i, j, 2] = kap_dist
		distance[i, j, 3] = h_dist
		distance[i, j, 4] = diam_dist
		distance[i, j, 5] = ax_dist
		
		distance[j, i, 0] = R_dist
		distance[j, i, 1] = th_dist
		distance[j, i, 2] = kap_dist
		distance[j, i, 3] = h_dist
		distance[j, i, 4] = diam_dist
		distance[j, i, 5] = ax_dist
		
	collect_mp.append(distance)
	
def calc_distances(profiles):
	# profiles[sample_id] = [profile, radius, params]
	
	profiles_n = len(profiles)
	sample_ids = list(profiles.keys())
	
	arc_lengths = {}
	tangents = {}
	curvatures = {}
	for sample_id in profiles:
		prof = profiles[sample_id][0] + [profiles[sample_id][1], 0]
		arc_lengths[sample_id] = arc_length(prof)
		tangents[sample_id] = fftsmooth(tangent(prof))
		curvatures[sample_id] = np.hstack(([0], np.diff(tangents[sample_id])))
	
	manager = mp.Manager()
	ijs_mp = manager.list(combinations(range(profiles_n), 2))
	collect_mp = manager.list()
	
	procs = []
	for pi in range(mp.cpu_count()):
		procs.append(mp.Process(target = dist_worker, args = (ijs_mp, collect_mp, profiles, sample_ids, arc_lengths, tangents, curvatures)))
		procs[-1].start()
	for proc in procs:
		proc.join()
	for proc in procs:
		proc.terminate()
		proc = None
	
	distance = np.ones((profiles_n, profiles_n, 6), dtype = float) # distance[i, j] = [R_dist, th_dist, kap_dist, h_dist, diam_dist, ax_dist], ; where i, j = indices in sample_ids; R_dist, th_dist, kap_dist, h_dist, diam_dist, ax_dists = components of morphometric distance
	for i in range(profiles_n):
		distance[i,i,:] = 0
	
	for dist in collect_mp:
		mask = (dist != -2)
		distance[mask] = dist[mask]
	
	return distance

def combine_dists(distance, w_R, w_th, w_kap, w_h, w_diam, w_ax):
	# distance[i, j] = [R_dist, th_dist, kap_dist, h_dist, diam_dist, ax_dist]; where i, j = indices in sample_ids; R_dist, th_dist, kap_dist, h_dist, diam_dist, ax_dist = components of morphometric distance
	
	combined = np.ones(distance.shape[:2])
	
	w_sum = sum([w_R, w_th, w_kap, w_h, w_diam, w_ax])
	w_R, w_th, w_kap, w_h, w_diam, w_ax = [w / w_sum for w in [w_R, w_th, w_kap, w_h, w_diam, w_ax]]
	
	dists = [None] * 6
	for idx in range(6):
		dists[idx] = distance[:,:,idx]
	R_dist, th_dist, kap_dist, h_dist, diam_dist, ax_dist = dists
	
	ax_dist[ax_dist == -1] = ax_dist[ax_dist > -1].mean()
	
	combined = R_dist * w_R + th_dist * w_th + kap_dist * w_kap + h_dist * w_h + diam_dist * w_diam + ax_dist * w_ax
	
	for i in range(combined.shape[0]):
		combined[i,i] = 0
	
	return combined

def combine_dists_rms(distance, w_R, w_th, w_kap, w_h, w_diam, w_ax):
	# distance[i, j] = [R_dist, th_dist, kap_dist, h_dist, diam_dist, ax_dist]; where i, j = indices in sample_ids; R_dist, th_dist, kap_dist, h_dist, diam_dist, ax_dist = components of morphometric distance
	
	combined = np.ones(distance.shape[:2])
	
	w_sum = sum([w_R, w_th, w_kap, w_h, w_diam, w_ax])
	w_R, w_th, w_kap, w_h, w_diam, w_ax = [w / w_sum for w in [w_R, w_th, w_kap, w_h, w_diam, w_ax]]
	
	dists = [None] * 6
	for idx in range(6):
		dists[idx] = distance[:,:,idx]
	R_dist, th_dist, kap_dist, h_dist, diam_dist, ax_dist = dists
	
	ax_dist[ax_dist == -1] = ax_dist[ax_dist > -1].mean()
	
	combined = np.sqrt((R_dist**2) * w_R + (th_dist**2) * w_th + (kap_dist**2) * w_kap + (h_dist**2) * w_h + (diam_dist**2) * w_diam + (ax_dist**2) * w_ax)
	
	for i in range(combined.shape[0]):
		combined[i,i] = 0
	
	return combined

def get_dendrogram(z):
	# return [links, xticks, xlabels]
	# links: [[k, x, y, link, leaves], ...]; link = [[x,y], ...]; leaves = list of leaves connected by link

	def getLeaves(z, k):
		leaves = []
		n = z.shape[0] + 1
		left, right = z[k,:2]
		if left < n:
			leaves.append(int(left))
		else:
			leaves += getLeaves(z, int(left - n))
		if right < n:
			leaves.append(int(right))
		else:
			leaves += getLeaves(z, int(right - n))
		return leaves
	
	n = z.shape[0] + 1
	ddata = dendrogram(z, link_color_func = lambda k: "#000000"[:-len(str(k))] + str(k), no_plot = True)
	links = [[int(ddata["color_list"][k][1:]) - n, 0.5 * sum(ddata['icoord'][k][1:3]), ddata['dcoord'][k][1], np.vstack((ddata['icoord'][k], ddata['dcoord'][k])).T, []] for k in range(len(ddata['icoord']))]
	xticks = np.zeros(n)
	for k, x, y, link, leaves in links:
		left, right = z[k,:2]
		if left < n:
			xticks[ddata["leaves"].index(int(left))] = link[0,0]
		if right < n:
			xticks[ddata["leaves"].index(int(right))] = link[3,0]

	for k, x, y, link, leaves in links:
		leaves += getLeaves(z, k)
	
	return links, xticks, ddata["leaves"]

def get_clusters(distance, iters = 1000):
	
	pca = PCA(n_components = None)
	pca.fit(distance)
	scree = pca.explained_variance_ratio_
	s = np.cumsum(scree)
	pca = PCA(n_components = int(((s <= 0.9) | (scree > 0.5)).sum()))
	pca.fit(distance)
	pca_scores = pca.transform(distance)
	z = linkage(pca_scores, method = "ward")
	
	# find optimal number of clusters based on within-cluster sum of squares
	lda = LinearDiscriminantAnalysis()
	fitness = []
	for n_clusters in range(2, distance.shape[0] // 2):
		clusters = np.array(fcluster(z, n_clusters, "maxclust"), dtype = int)
		stats = np.zeros(clusters.shape[0])
		for r in range(iters):
			mask = np.random.randint(0, 2, clusters.shape[0]).astype(bool)
			lda.fit(pca_scores[mask], clusters[mask])
			pred = lda.predict(pca_scores[~mask])
			stats[~mask] += (pred == clusters[~mask]).astype(int)
		n_clusters = clusters.max()
		fitness.append([clusters.max(), ((n_clusters*stats)/(n_clusters*stats + iters)).mean()])
	fitness = np.array(fitness)
	fitness = fitness[np.argsort(fitness[:,0])]
	idx = np.argmax(fitness[:,1])
	n_clusters = int(fitness[idx,0])
	fitness_opt = fitness[idx,1]
	
	# get cluster names
	names = {} # {k: name, ...}
	links, _, _ = get_dendrogram(z)
	parent = {} #  {k_child: k_parent, ...}
	n = z.shape[0] + 1
	k_leaves = {} # {k: leaves, ...}
	ks = []
	for k, _, _, _, leaves in links:
		k_leaves[k] = leaves.copy()
		left, right = z[k,:2].astype(int) - n
		parent[left] = k
		parent[right] = k
		ks.append(k)
	ks = sorted(ks)[::-1]
	for k in ks:
		name = str(ks[0] - k)
		if k in parent:
			if names[parent[k]] != "0":
				names[k] = "%s.%s" % (names[parent[k]], name)
			else:
				names[k] = name
		else:
			names[k] = "0"
	
	# assign names to clusters
	clusters = np.array(fcluster(z, n_clusters, "maxclust"), dtype = int)
	clu_labels = np.unique(clusters)
	collect = {}
	for clu in clu_labels:
		cluster = np.where(clusters == clu)[0]
		if cluster.size:
			for k in k_leaves:
				if (len(k_leaves[k]) == len(cluster)) and np.in1d(cluster, k_leaves[k]).all():
					break
			collect[names[k]] = cluster.tolist()
	clu_names = sorted(collect.keys())
	collect = dict([(name, collect[name]) for name in clu_names])
	
	return collect, fitness_opt

def get_clusters_wss(distance):
	
	pca = PCA(n_components = None)
	pca.fit(distance)
	scree = pca.explained_variance_ratio_
	s = np.cumsum(scree)
	pca = PCA(n_components = int(((s <= 0.9) | (scree > 0.5)).sum()))
	pca.fit(distance)
	pca_scores = pca.transform(distance)
	z = linkage(pca_scores, method = "ward")
	
	# find optimal number of clusters based on within-cluster sum of squares
	wss = []
	for n_clusters in range(2, distance.shape[0] // 2):
		clusters = np.array(fcluster(z, n_clusters, "maxclust"), dtype = int)
		c_sum = 0
		for clu in range(1, clusters.max() + 1):
			mask = (clusters == clu)
			c_sum += (distance[:,mask][mask]**2).sum()
		wss.append([clusters.max(), c_sum])
	wss = np.array(wss)
	wss = wss[np.argsort(wss[:,0])]
	idx = np.where(np.gradient(wss[:,1]) > -1)[0].min()
	n_clusters = int(wss[idx,0])
	wss_opt = wss[idx,1]
	
	# get cluster names
	names = {} # {k: name, ...}
	links, _, _ = get_dendrogram(z)
	parent = {} #  {k_child: k_parent, ...}
	n = z.shape[0] + 1
	k_leaves = {} # {k: leaves, ...}
	ks = []
	for k, _, _, _, leaves in links:
		k_leaves[k] = leaves.copy()
		left, right = z[k,:2].astype(int) - n
		parent[left] = k
		parent[right] = k
		ks.append(k)
	ks = sorted(ks)[::-1]
	for k in ks:
		name = str(ks[0] - k)
		if k in parent:
			if names[parent[k]] != "0":
				names[k] = "%s.%s" % (names[parent[k]], name)
			else:
				names[k] = name
		else:
			names[k] = "0"
	
	# assign names to clusters
	clusters = np.array(fcluster(z, n_clusters, "maxclust"), dtype = int)
	clu_labels = np.unique(clusters)
	collect = {}
	for clu in clu_labels:
		cluster = np.where(clusters == clu)[0]
		if cluster.size:
			for k in k_leaves:
				if (len(k_leaves[k]) == len(cluster)) and np.in1d(cluster, k_leaves[k]).all():
					break
			collect[names[k]] = cluster.tolist()
	clu_names = sorted(collect.keys())
	collect = dict([(name, collect[name]) for name in clu_names])
	
	return collect, wss_opt

def get_max_clusters(distance):
	
	pca = PCA(n_components = None)
	pca.fit(distance)
	scree = pca.explained_variance_ratio_
	s = np.cumsum(scree)
	pca = PCA(n_components = int(((s <= 0.9) | (scree > 0.5)).sum()))
	pca.fit(distance)
	pca_scores = pca.transform(distance)
	
	z = linkage(pca_scores, method = "ward")
	
	# get cluster names
	names = {} # {k: name, ...}
	links, _, _ = get_dendrogram(z)
	parent = {} #  {k_child: k_parent, ...}
	n = z.shape[0] + 1
	k_leaves = {} # {k: leaves, ...}
	ks = []
	for k, _, _, _, leaves in links:
		k_leaves[k] = leaves.copy()
		left, right = z[k,:2].astype(int) - n
		parent[left] = k
		parent[right] = k
		ks.append(k)
	ks = sorted(ks)[::-1]
	for k in ks:
		name = str(ks[0] - k)
		if k in parent:
			if names[parent[k]] != "0":
				names[k] = "%s.%s" % (names[parent[k]], name)
			else:
				names[k] = name
		else:
			names[k] = "0"
	
	ks = sorted(names.keys(), key = lambda k: len(k_leaves[k]))
	
	clusters = np.array(fcluster(z, distance.shape[0] // 2, "maxclust"), dtype = int)
	clu_labels = np.unique(clusters)
	collect = defaultdict(list)
	for clu in clu_labels:
		cluster = np.where(clusters == clu)[0]
		if cluster.size:
			for k in ks:
				if np.in1d(cluster, k_leaves[k]).all():
					break
			collect[names[k]] += cluster.tolist()
	collect = dict([(name, collect[name]) for name in natsorted(collect.keys())])
	return collect

