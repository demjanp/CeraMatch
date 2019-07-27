import numpy as np
import multiprocessing as mp
from natsort import natsorted
from PIL import Image, ImageDraw
from itertools import combinations, product
from collections import defaultdict
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist, pdist
from scipy.cluster.hierarchy import linkage
from scipy.cluster.vq import kmeans,vq
from scipy import stats

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

def profile_axis(profile, params, step = 0.2, inner_weight = 0.5):
	# step: generate axis point every step mm
	# inner_weight: weight of the inner profile when calculating the axis
	
	idx_rim = np.argmin(cdist([get_rim(profile)], profile)[0])
	profile = fftsmooth(profile, 0.1, params)
	inner = profile[:idx_rim][::-1]
	outer = profile[idx_rim:]
	d_inner = np.sqrt((inner**2).sum(axis = 1))
	d_outer = np.sqrt((outer**2).sum(axis = 1))
	d_max = min(d_inner.max(), d_outer.max())
	d_min = max(d_inner[d_inner > 0].min(), d_outer[d_outer > 0].min())
	collect = [[0,0]]
	for d in np.arange(d_min, d_max, step):
		collect.append((inner[d_inner > d][0]*inner_weight + outer[d_outer > d][0]*(1 - inner_weight)))
		if cdist([collect[-1]], [collect[-2]])[0][0] < step:
			del collect[-1]
	return np.array(collect)

def rotate_profile(profile, angle):
	
	ca = np.cos(angle)
	sa = np.sin(angle)
	rot_matrix = np.array([[ca, sa],[-sa, ca]])
	return np.dot(profile, rot_matrix.T)

def diameter_dist(radius1, radius2):
	
	r1, r2 = sorted([radius1, radius2])
	return 1 - r1 / r2

def thickness_dist(thickness1, thickness2):
	
	t1, t2 = sorted([thickness1, thickness2])
	return 1 - t1 / t2

def hamming_dist(prof1, prof2, rasterize_factor = 10):
	# prof1, prof2: [[x,y], ...]
	# prof1 & 2 are assumed to be shifted to 0 ([0,0] coordinate is at the rim)
	'''
		H = 1 - (2*Ao) / (A1 + A2)
		H = hamming distance <0..1>
		Ao = area of overlaping of profiles
		A1, A2 = areas of profiles
	'''
	
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

def rotated_hamming_dist(prof1, axis1, thickness1, prof2, axis2, thickness2, rasterize_factor = 10, trim_by_thickness = True):
	
	collect = []
	for prof, axis in [[prof1, axis1], [prof2, axis2]]:
		
		x1, y1 = axis[np.argmin(axis[:,1])]
		x2, y2 = axis[np.argmax(axis[:,1])]
		slope = (y2 - y1) / (x2 - x1)
		angle = np.arctan(slope)
		
		if angle > 0:
			angle = angle + np.pi
		angle += np.pi / 2
		prof = rotate_profile(prof, angle)
		collect.append(prof - get_rim(prof))
	prof1, prof2 = collect
	
	if trim_by_thickness:
		y_max = min(3*min(thickness1, thickness2), prof1[:,1].max(), prof2[:,1].max())
	else:
		y_max = min(prof1[:,1].max(), prof2[:,1].max())
	prof1 = prof1[(prof1[:,1] < y_max)]
	prof2 = prof2[(prof2[:,1] < y_max)]
	
	idx_maxd = np.argmax(prof1[:,1])
	rot_step = 0.016
	while True:
		prof_r = rotate_profile(prof1, rot_step)
		dif = ((prof1[idx_maxd] - prof_r[idx_maxd])**2).sum()
		if dif > 0.5:
			rot_step *= 0.9
		elif dif < 0.4:
			rot_step *= 1.2
		else:
			break
	
	angle_min = -0.2
	angle_max = 0.2
	angle_opt = None
	h_dist_min = np.inf
	for angle in np.linspace(angle_min, angle_max, (angle_max - angle_min) / rot_step):
		prof1_r = rotate_profile(prof1, angle)
		prof1_r -= get_rim(prof1_r)
		h_dist = hamming_dist(prof1_r, prof2, rasterize_factor = rasterize_factor)
		if h_dist < h_dist_min:
			h_dist_min = h_dist
			angle_opt = angle
	prof1 = rotate_profile(prof1, angle_opt)
	prof1 -= get_rim(prof1)
	
	shifts1 = prof1[(prof1[:,1] == prof1[:,1].min())][:,0]
	shifts2 = prof2[(prof2[:,1] == prof2[:,1].min())][:,0]
	h_dist = np.inf
	for shift1 in shifts1:
		for shift2 in shifts2:
			h_dist = min(h_dist, hamming_dist(prof1 - [shift1, 0], prof2 - [shift2, 0], rasterize_factor = rasterize_factor))
	return h_dist

def axis_dist(axis1, axis2, axis_step):
	
	axis1, xd1, yd1, length1 = axis1
	axis2, xd2, yd2, length2 = axis2
	
	break_len = 0
	
	if length1 < length2:
		axis_shorter = axis1
		axis_longer = axis2
		if xd1 is not None:
			length = length2 - length1
			break1 = np.linspace(0, length * axis_step, int(round(length / axis_step)))
			break1 = np.vstack((break1*xd1, break1*yd1)).T + axis1[-1]
			axis_shorter = np.vstack((axis_shorter, break1))
			break_len = break1.shape[0]
	
	elif length2 < length1:
		axis_shorter = axis2
		axis_longer = axis1
		if xd2 is not None:
			length = length1 - length2
			break2 = np.linspace(0, length * axis_step, int(round(length / axis_step)))
			break2 = np.vstack((break2*xd2, break2*yd2)).T + axis2[-1]
			axis_shorter = np.vstack((axis2, break2))
			break_len = break2.shape[0]
	
	d_norm = np.abs(np.linspace(0, np.sqrt((axis_longer[-1]**2).sum()), axis_shorter.shape[0]) + np.linspace(0, np.sqrt((axis_shorter[-1]**2).sum()), axis_shorter.shape[0]))
	if break_len > 0:
		w = np.hstack((np.ones(axis_shorter.shape[0] - break_len), np.linspace(1, 0, break_len)))
		w = w / w.sum()
		d_norm = (d_norm * w).sum()
	else:
		d_norm = d_norm.mean()
		w = np.ones(axis_shorter.shape[0]) / axis_shorter.shape[0]
	d1 = cdist(axis_shorter, axis_longer).min(axis = 1)
	d2 = cdist(axis_longer, axis_shorter).min(axis = 1)
	
	d_norm2 = np.abs(np.linspace(0, np.sqrt((axis_shorter[-1]**2).sum()), axis_longer.shape[0]) + np.linspace(0, np.sqrt((axis_longer[-1]**2).sum()), axis_longer.shape[0])).mean()
	d2_sum = 0
	for pnt in axis_longer:
		d = cdist([pnt], axis_shorter)[0]
		idx = np.argmin(d)
		d2_sum += d[idx] * w[idx]
	
	return np.sqrt((((d1 * w).sum() / d_norm) + (d2_sum / d_norm2)) / 2)

def arc_length(profile):
	
	dx = np.gradient(profile[:,0])
	dy = np.gradient(profile[:,1])
	s = np.cumsum(np.sqrt(dx**2 + dy**2))
	s -= s[np.argmin(cdist([get_rim(profile)], profile)[0])]
	return s

def tangent(profile):
	
	dx = np.gradient(profile[:,0])
	dy = np.gradient(profile[:,1])
	th = np.zeros(profile.shape[0])
	mask = (dx != 0)
	th[mask] = (dy[mask] / dx[mask])
	return th

def fftsmooth(profile, threshold = 0.2, params = None):
	
	if profile.shape[0] < 5:
		return profile
	
	# extend profile at break point by 100 mm if params are available
	if params:
		_, _, xd, yd = params
		step = 0.1
		line = np.vstack((xd * np.arange(0, 100, step), yd * np.arange(0, 100, step))).T
		profile = np.vstack((profile[0] + line[::-1], profile, profile[-1] + line))
	
	# make sure profile line is closed
	pnt0 = (profile[0] + profile[-1]) / 2
	profile = np.vstack(([pnt0], profile, [pnt0]))
	
	fft = np.fft.fft2(profile)
	freq = np.abs(np.fft.fftfreq(profile.shape[0]))
	fft[np.abs(freq) > threshold] = 0
	profile = np.fft.ifft2(fft)[1:-1].real
	
	if params:
		profile = profile[line.shape[0]:-line.shape[0]]
	
	return profile

def interpolate_signal(s, signal, s_min, s_max, resolution = 0.1):
	
	xs = np.arange(s_min, s_max + resolution, resolution)
	return xs, np.interp(xs, s, signal)

def radius_dist(prof1, prof2):
	
	R1 = prof1[:,0]
	R2 = prof2[:,0]
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

def dist_worker_alt(ijs_mp, collect_mp, profiles, sample_ids, arc_lengths, tangents, curvatures):
	
	profiles_n = len(profiles)
	distance = np.zeros((profiles_n, profiles_n, 3), dtype = float) - 2
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
		
		profile1, _, radius1, _, _ = profiles[sample_ids[i]]
		profile2, _, radius2, _, _ = profiles[sample_ids[j]]
		
		R_dist = np.inf
		shifts1 = profile1[(profile1[:,1] == profile1[:,1].min())][:,0]
		shifts2 = profile2[(profile2[:,1] == profile2[:,1].min())][:,0]
		for shift1 in shifts1:
			for shift2 in shifts2:		
				R_dist = min(R_dist, radius_dist(profile1 - [shift1, 0] + [radius1, 0], profile2 - [shift2, 0] + [radius2, 0]))
		
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
		
		distance[j, i, 0] = R_dist
		distance[j, i, 1] = th_dist
		distance[j, i, 2] = kap_dist
		
	collect_mp.append(distance)

def calc_distances_alt(profiles):
	# profiles[sample_id] = [profile, bottom, radius, thickness, params]
	
	profiles_n = len(profiles)
	sample_ids = list(profiles.keys())
	
	arc_lengths = {}
	tangents = {}
	curvatures = {}
	for sample_id in profiles:
		profile, bottom, radius, _, _ = profiles[sample_id]
		
		# calculate arc_length, tangent, curvature
		prof = profile + [radius, 0]
		arc_lengths[sample_id] = arc_length(prof)
		tangents[sample_id] = tangent(prof)
		curvatures[sample_id] = np.gradient(tangents[sample_id])
		
	manager = mp.Manager()
	ijs_mp = manager.list(list(combinations(range(profiles_n), 2)))
	collect_mp = manager.list()
	
	procs = []
	for pi in range(mp.cpu_count()):
		procs.append(mp.Process(target = dist_worker_alt, args = (ijs_mp, collect_mp, profiles, sample_ids, arc_lengths, tangents, curvatures)))
		procs[-1].start()
	for proc in procs:
		proc.join()
	for proc in procs:
		proc.terminate()
		proc = None
	
	distance = np.ones((profiles_n, profiles_n, 3), dtype = float) # distance[i, j] = [R_dist, th_dist, kap_dist], ; where i, j = indices in sample_ids; R_dist, th_dist, kap_dist = components of morphometric distance
	for i in range(profiles_n):
		distance[i,i,:] = 0
	
	for dist in collect_mp:
		mask = (dist != -2)
		distance[mask] = dist[mask]
	
	return distance

def dist_worker(ijs_mp, collect_mp, profiles, sample_ids, thicknesses, axes, axis_step):
	
	profiles_n = len(profiles)
	distance = np.zeros((profiles_n, profiles_n, 5), dtype = float) - 2
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
		
		profile1, _, radius1, _, _ = profiles[sample_ids[i]]
		profile2, _, radius2, _, _ = profiles[sample_ids[j]]
		
		diam_dist = diameter_dist(radius1, radius2)
		thick_dist = thickness_dist(thicknesses[sample_ids[i]], thicknesses[sample_ids[j]])
		ax_dist = axis_dist(axes[sample_ids[i]], axes[sample_ids[j]], axis_step)
		h_dist = rotated_hamming_dist(profile1, axes[sample_ids[i]][0], 0, profile2, axes[sample_ids[j]][0], 0, trim_by_thickness = False)
		ht_dist = rotated_hamming_dist(profile1, axes[sample_ids[i]][0], thicknesses[sample_ids[i]], profile2, axes[sample_ids[j]][0], thicknesses[sample_ids[j]], trim_by_thickness = True)
		
		distance[i, j, 0] = diam_dist
		distance[i, j, 1] = thick_dist
		distance[i, j, 2] = ax_dist
		distance[i, j, 3] = h_dist
		distance[i, j, 4] = ht_dist
		
		distance[j, i, 0] = diam_dist
		distance[j, i, 1] = thick_dist
		distance[j, i, 2] = ax_dist
		distance[j, i, 3] = h_dist
		distance[j, i, 4] = ht_dist
		
	collect_mp.append(distance)
	
def calc_distances(profiles):
	# profiles[sample_id] = [profile, bottom, radius, thickness, params]
	
	axis_step = 0.5
	
	profiles_n = len(profiles)
	sample_ids = list(profiles.keys())
	
	axes = {}  # {sample_id: [axis, xd, yd], ...}
	thicknesses = {} # {sample_id: thickness, ...}
	for sample_id in profiles:
		profile, bottom, radius, thickness, params = profiles[sample_id]
		
		thicknesses[sample_id] = thickness
		
		# calculate axis
		axis = profile_axis(profile, params, step = axis_step, inner_weight = 0.5)
#		if bottom:
#			mask = (axis[:,1] < profile[:,1].max() - thickness)
#			if mask.any():
#				axis = axis[mask]
		length = np.sqrt((axis[-1]**2).sum())
		xd, yd = None, None
		if params:
			_, _, xd, yd = params
		axes[sample_id] = [axis, xd, yd, length]
		
	manager = mp.Manager()
	ijs_mp = manager.list(list(combinations(range(profiles_n), 2)))
	collect_mp = manager.list()
	
	procs = []
	for pi in range(mp.cpu_count()):
		procs.append(mp.Process(target = dist_worker, args = (ijs_mp, collect_mp, profiles, sample_ids, thicknesses, axes, axis_step)))
		procs[-1].start()
	for proc in procs:
		proc.join()
	for proc in procs:
		proc.terminate()
		proc = None
	
	distance = np.ones((profiles_n, profiles_n, 5), dtype = float) # distance[i, j] = [diam_dist, thick_dist, ax_dist, h_dist, ht_dist], ; where i, j = indices in sample_ids; [name]_dist = components of morphometric distance
	for i in range(profiles_n):
		distance[i,i,:] = 0
	
	for dist in collect_mp:
		mask = (dist != -2)
		distance[mask] = dist[mask]
	
	return distance

def calc_pca_scores(D, n_components = None):
	
	if n_components is None:
		pca = PCA(n_components = None)
		pca.fit(D)
		n_components = np.where(np.cumsum(pca.explained_variance_ratio_) > 0.99)[0]
		if not n_components.size:
			n_components = 1
		else:
			n_components = n_components.min() + 1
	pca = PCA(n_components = n_components)
	pca.fit(D)
	return pca.transform(D)

def calc_mode(D):
	
	d = D.flatten()
	if d.shape[0] == 1:
		return d[0]
	d = d[d > 0]
	return stats.mode(np.round(d, 3))[0][0]

def combine_dists(distance, weights):
	# weights = [w_dist1, w_dist2, w_dist3]
	# distance[i, j] = [dist1, dist2, dist3]; where i, j = indices in sample_ids; dist1, dist2, dist3 = components of morphometric distance
	
	combined = np.ones(distance.shape[:2])
	
	w_sum = sum(weights)
	if w_sum == 0:
		weights = [1/6] * len(weights)
	else:
		weights = [w / w_sum for w in weights]
	
	dists = [None] * 3
	for idx in range(3):
		dists[idx] = distance[:,:,idx]
	
	combined = dists[0] * weights[0] + dists[1] * weights[1] + dists[2] * weights[2]
	
	for i in range(combined.shape[0]):
		combined[i,i] = 0
	
	return combined

def get_distmax_ordering(pca_scores, D):
	# orders samples so that the the next one has the highest possible distance to the most similar of the previous samples
	# (ensures that the samples at the beginning of the sequence are the most diverse)
	# returns {sample_idx: order, ...}
	
	# start with the profile most disstant from the centroid of the assemblage
	idx0 = np.argmax(((pca_scores - pca_scores.mean(axis = 0))**2).sum(axis = 1))
	idxs_done = [idx0]
	data = {} # {sample_idx: order, ...}
	i = 0
	data[idx0] = i
	idxs_todo = [idx for idx in range(D.shape[0]) if idx != idx0]
	while idxs_todo:
		i += 1
		idx1 = idxs_todo[np.argmax((D[:,idxs_todo][idxs_done]).min(axis = 0))]
		d = D[idxs_done,idx1].min()
		data[idx1] = i
		idxs_done.append(idx1)
		idxs_todo.remove(idx1)
	return data

def get_sample_labels(distance):
	# returns {sample_idx: label, ...}
	
	samples_n = distance.shape[0]
	z = linkage(distance, method = "ward") # [[idx1, idx2, dist, sample_count], ...]
	dist_max = z[:,3].max()
	sample_idxs = {} # {z_idx: [sample_idx, ...], ...}
	dist_lookup = {} # {z_idx: dist, ...}
	z_idx = samples_n
	for idx1, idx2, dist in z[:,:3]:
		if idx1 < samples_n:
			idxs1 = [int(idx1)]
		else:
			idxs1 = sample_idxs[idx1]
		if idx2 < samples_n:
			idxs2 = [int(idx2)]
		else:
			idxs2 = sample_idxs[idx2]
		sample_idxs[z_idx] = idxs1 + idxs2
		dist_lookup[z_idx] = dist_max - dist
		z_idx += 1
	z_idx_max = max(sample_idxs.keys())
	sample_idxs = dict([(z_idx_max - z_idx, sample_idxs[z_idx]) for z_idx in sample_idxs])
	sample_idxs = dict([(z_idx, sample_idxs[z_idx]) for z_idx in sorted(sample_idxs.keys())])
	
	idx_max = max(sample_idxs.keys())
	for idx in range(samples_n):
		sample_idxs[idx_max + idx + 1] = [idx]
	
	sample_labels = defaultdict(str) # {sample_idx: label, ...}
	for z_idx in sorted(sample_idxs.keys()):
		for idx in sample_idxs[z_idx]:
			sample_labels[idx] += ".%d" % (int(z_idx))
	for idx in sample_labels:
		sample_labels[idx] = ".".join(sample_labels[idx].split(".")[2:])
	
	return sample_labels

def calc_ssd(pca_scores):
	# calculate sum of squared distances
	
	return (((pca_scores - pca_scores.mean(axis = 0))**2).sum(axis = 1)).sum()

def get_2_clusters(pca_scores):
	
	if pca_scores.shape[0] <= 2:
		return [np.array([i], dtype = int) for i in range(pca_scores.shape[0])], 1.0
	clusters_l, _ = vq(pca_scores, kmeans(pca_scores, 2)[0])
	labels = np.unique(clusters_l)
	clusters = []
	for label in labels:
		clusters.append(np.where(clusters_l == label)[0].astype(int))
	if len(clusters) != 2:
		return clusters, 1.0
	ssd_samples = calc_ssd(pca_scores)
	ssd_clusters = 0
	for cluster in clusters:
		if len(cluster) < 2:
			pass
		else:
			ssd_clusters += calc_ssd(pca_scores[cluster])
	ci = ssd_clusters / ssd_samples
	return clusters, ci

def get_ci_mp(params):
	
	weights, D_samples = params
	_, ci = get_2_clusters(calc_pca_scores(combine_dists(D_samples, weights)))
	return ci

def get_cis_sp(weights, D_samples):
	
	cis = []
	for weights_case in weights:
		_, ci = get_2_clusters(calc_pca_scores(combine_dists(D_samples, weights_case)))
		cis.append(ci)
	return np.array(cis)

def split_cluster(D, samples, weights, pool):
	# returns [cluster, ...]
	
	if len(samples) <= 2:
		return [np.array([idx], dtype = int) for idx in samples], 1.0
	
	D_samples = D[:,samples][samples]
	if len(samples) > 100:
		cis = np.array(pool.map(get_ci_mp, ([weights_case, D_samples] for weights_case in weights)))
	else:
		cis = get_cis_sp(weights, D_samples)
	
	weights_case = weights[np.argmin(cis)]
	pca_scores = calc_pca_scores(combine_dists(D[:,samples][samples], weights_case))
	clusters, ci_obs = get_2_clusters(pca_scores)
	
	samples = np.array(samples, dtype = int)
	return [samples[cluster] for cluster in clusters], weights_case, ci_obs
	
def get_auto_clusters(D, limits = None):
	
	def _get_max_cube_idx(same):
		
		for idx in range(same.shape[0]):
			if not same[:,:idx][:idx].all():
				return max(0, idx - 1)
		return idx
	
	def _get_max_cluster(same, samples, pca_1):
		
		same = same[:,samples][samples]
		if same.all():
			return samples
		
		try:
			pca_1.fit(~same)
		except:
			print("Warning! PCA error. Matrix:")
			print(same.astype(int))
			return samples
		order1 = np.argsort(pca_1.transform(~same)[:,0])
		order2 = order1[::-1]
		same1 = same[:,order1][order1]
		same2 = same[:,order2][order2]
		idx1 = _get_max_cube_idx(same1)
		idx2 = _get_max_cube_idx(same2)
		if idx1 > idx2:
			return samples[order1[:idx1]]
		else:
			return samples[order2[:idx2]]
	
	def _get_clusters_pure(same):
		
		samples = np.arange(same.shape[0], dtype = int)
		
		# succesively collect clusters by finding the largest first, than the largest from the remaining samples etc.
		# i.e. collect the samples into clusters that don't overlap maximizing the average size of the clusters
		clusters = []
		pca_1 = PCA(n_components = 1)
		while samples.size:
			cluster = _get_max_cluster(same, samples, pca_1)
			clusters.append(cluster)
			samples = np.setdiff1d(samples, cluster)
		order_idxs = np.argsort([len(cluster) for cluster in clusters])[::-1]
		clusters = [sorted(clusters[ci], key = lambda idx: order_idxs[ci])[::-1] for ci in order_idxs]
		
		clusters = dict([(str(i + 1), cluster) for i, cluster in enumerate(clusters)])
		return clusters
	
	if limits is None:
		limits = []
		for di in range(D.shape[2]):
			limits.append(calc_mode(D[:,:,di]))
	D = D / limits
	same = (D < 1).all(axis = 2)
	clusters = _get_clusters_pure(same)
	collect = {}
	singles = []
	outliers = []
	for label in clusters:
		if len(clusters[label]) < 2:
			singles += clusters[label]
			outliers += clusters[label]
			continue
		collect[label] = clusters[label].copy()
	clusters = collect
	labels = list(clusters.keys())
	keep = []
	to_add = []
	for idx in singles:
		rs = np.array([np.sqrt((D[idx,clusters[label]]**2).sum(axis = 1)).mean() for label in labels])
		r_min = rs.min()
		if r_min < 2:
			idxs_labels = np.where(rs == r_min)[0].tolist()
			label = labels[sorted(idxs_labels, key = lambda idx_label: len(clusters[labels[idx_label]]))[::-1][0]]
			to_add.append([label, idx])
		else:
			keep.append(idx)
	for label, idx in to_add:
		clusters[label] = np.hstack((clusters[label], [idx]))
	if keep:
		clusters["singles"] = np.array(keep, dtype = int)
	
	weights = {} # {label: limits, ...}
	collect = {}  # {sample_idx: label, ...}
	for label in clusters:
		weights[label] = limits[:3]  # DEBUG
		for sample_idx in clusters[label]:
			collect[sample_idx] = label
	return collect, outliers, weights

