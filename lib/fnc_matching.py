import numpy as np
import multiprocessing as mp
from natsort import natsorted
from PIL import Image, ImageDraw
from itertools import combinations
from collections import defaultdict
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist, pdist
from scipy.cluster.hierarchy import linkage, fcluster

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
	# step: generate axis point every stem mm
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
	# trim both profiles to the length of the shorter
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
	d = cdist(axis_shorter, axis_longer).min(axis = 1)
	return (d * w).sum() / d_norm

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

def dist_worker(ijs_mp, collect_mp, profiles, sample_ids, arc_lengths, tangents, curvatures, axes, axis_step):
	
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
		
		profile1, _, radius1, _, _ = profiles[sample_ids[i]]
		profile2, _, radius2, _, _ = profiles[sample_ids[j]]
		
		shifts1 = profile1[(profile1[:,1] == profile1[:,1].min())][:,0]
		shifts2 = profile2[(profile2[:,1] == profile2[:,1].min())][:,0]
		
		h_dist = np.inf
		ax_dist = np.inf
		R_dist = np.inf
		
		for shift1 in shifts1:
			for shift2 in shifts2:
				h_dist = min(h_dist, hamming_dist(profile1 - [shift1, 0], profile2 - [shift2, 0]))
		
		shifts1 = profile1[(profile1[:,1] == profile1[:,1].min())][:,0]
		shifts2 = profile2[(profile2[:,1] == profile2[:,1].min())][:,0]
		for shift1 in shifts1:
			for shift2 in shifts2:		
				R_dist = min(R_dist, radius_dist(profile1 - [shift1, 0] + [radius1, 0], profile2 - [shift2, 0] + [radius2, 0]))
		
		diam_dist = diameter_dist(radius1, radius2)
		
		axis1 = axes[sample_ids[i]] # [axis, xd, yd, length]
		axis2 = axes[sample_ids[j]]
		
		ax_dist = axis_dist(axis1, axis2, axis_step)
		
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
	# profiles[sample_id] = [profile, bottom, radius, thickness, params]
	
	axis_step = 1
	
	profiles_n = len(profiles)
	sample_ids = list(profiles.keys())
	
	arc_lengths = {}
	tangents = {}
	curvatures = {}
	axes = {}  # {sample_id: [axis, xd, yd], ...}
	for sample_id in profiles:
		profile, bottom, radius, thickness, params = profiles[sample_id]
		
		# calculate arc_length, tangent, curvature
		prof = profile + [radius, 0]
		arc_lengths[sample_id] = arc_length(prof)
		tangents[sample_id] = tangent(prof)
		curvatures[sample_id] = np.gradient(tangents[sample_id])
		
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
		procs.append(mp.Process(target = dist_worker, args = (ijs_mp, collect_mp, profiles, sample_ids, arc_lengths, tangents, curvatures, axes, axis_step)))
		procs[-1].start()
	for proc in procs:
		proc.join()
	for proc in procs:
		proc.terminate()
		proc = None
	
	distance = np.ones((profiles_n, profiles_n, 6), dtype = float) # distance[i, j] = [R_dist, th_dist, kap_dist, h_dist, diam_dist, axis_dist], ; where i, j = indices in sample_ids; R_dist, th_dist, kap_dist, h_dist, diam_dist, axis_dist = components of morphometric distance
	for i in range(profiles_n):
		distance[i,i,:] = 0
	
	for dist in collect_mp:
		mask = (dist != -2)
		distance[mask] = dist[mask]
	
	return distance

def combine_dists(distance, w_R, w_th, w_kap, w_h, w_diam, w_axis):
	# distance[i, j] = [R_dist, th_dist, kap_dist, h_dist, diam_dist, axis_dist]; where i, j = indices in sample_ids; R_dist, th_dist, kap_dist, h_dist, diam_dist, axis_dist = components of morphometric distance
	
	combined = np.ones(distance.shape[:2])
	
	w_sum = sum([w_R, w_th, w_kap, w_h, w_diam, w_axis])
	if w_sum == 0:
		w_R, w_th, w_kap, w_h, w_diam, w_axis = [1 / 6] * 6
	else:
		w_R, w_th, w_kap, w_h, w_diam, w_axis = [w / w_sum for w in [w_R, w_th, w_kap, w_h, w_diam, w_axis]]
	
	dists = [None] * 6
	for idx in range(6):
		dists[idx] = distance[:,:,idx]
	R_dist, th_dist, kap_dist, h_dist, diam_dist, axis_dist = dists
	
	combined = R_dist * w_R + th_dist * w_th + kap_dist * w_kap + h_dist * w_h + diam_dist * w_diam + axis_dist * w_axis
	
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

