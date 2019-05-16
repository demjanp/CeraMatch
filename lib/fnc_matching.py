import numpy as np
import multiprocessing as mp
from natsort import natsorted
from PIL import Image, ImageDraw
from itertools import combinations
from collections import defaultdict
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist, pdist
from scipy.spatial.distance import squareform
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

def landmark_dist(land1, land2):
	# land1, land2 = neck_vector, neck_length, upper_vector, upper_length, lower_vector, lower_length, break_vector
	
	neck_vector1, neck_length1, upper_vector1, upper_length1, lower_vector1, lower_length1, break_vector1 = land1
	neck_vector2, neck_length2, upper_vector2, upper_length2, lower_vector2, lower_length2, break_vector2 = land2
	
	mean_length1 = [length for length in [neck_length1, upper_length1, lower_length1] if length != 0]
	mean_length1 = sum(mean_length1) / len(mean_length1)
	mean_length2 = [length for length in [neck_length2, upper_length2, lower_length2] if length != 0]
	mean_length2 = sum(mean_length2) / len(mean_length2)
	
	dist = 0
	norm = 0
	for vector1, length1, vector2, length2 in [
		[neck_vector1, neck_length1, neck_vector2, neck_length2],
		[upper_vector1, upper_length1, upper_vector2, upper_length2],
		[lower_vector2, lower_length1, lower_vector2, lower_length2],
		[break_vector1, 0, break_vector2, 0],
	]:
		if length1 == 0:
			length1 = mean_length1
		if length2 == 0:
			length2 = mean_length2
		dist += np.sqrt(((vector1 - vector2)**2).sum())*length1*length2
		norm += 2*length1*length2
	return dist / norm

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

def get_landmarks(profile, bottom, thickness, params):
	# returns axis_start, concave, convex, axis_end; =[x, y]
	
	rim = get_rim(profile)
	
	profile = profile - rim
	
	axis_full = profile_axis(profile, params, inner_weight = 0.5)
	
	# crop axis to avoid issues with rim and bottom
	axis = axis_full[np.sqrt((axis_full**2).sum(axis = 1)) > 2*thickness]
	if not axis.size:
		axis = axis_full
	elif bottom:
		mask = (axis[:,1] < profile[:,1].max() - thickness)
		if mask.any():
			axis = axis[mask]
	
	dx, dy = axis[0] - axis[-1]
	angle_max = np.abs(np.arctan(dy/dx)) + np.pi / 2
	
	iters = 100
	# integrate y coordinates of the gradually rotated profile
	integrated = np.zeros(axis.shape[0])
	norm = np.zeros(axis.shape[0])
	straight = np.vstack((np.linspace(0, axis[-1,0], axis.shape[0]), np.linspace(0, axis[-1,1], axis.shape[0]))).T
	for angle in np.linspace(0, angle_max, iters):
		rotated = rotate_profile(axis, -angle)[:,1]
		integrated += (rotated - rotated.min())
		rotated_straight = rotate_profile(straight, -angle)[:,1]
		norm += (rotated_straight - rotated_straight.min())
	integrated = (integrated - norm) / iters
	
	concave = np.argmin(integrated)
	convex = np.argmax(integrated)
	
	axis_start = axis[0] + rim
	axis_end = axis[-1] + rim
	
	if cdist([axis[concave] + rim], [axis_start, axis_end])[0].min() < 0.5:
		concave = None
	
	if cdist([axis[convex] + rim], [axis_start, axis_end])[0].min() < 0.5:
		convex = None
	
	if (concave is not None) and (convex is not None) and (concave > convex):
		integrated = np.abs(integrated - integrated.mean())
		if integrated[concave] > integrated[convex]:
			convex = None
		else:
			concave = None
	
	if concave is not None:
		concave = axis[concave] + rim
	
	if convex is not None:
		convex = axis[convex] + rim
	
	if (concave is None) or cdist([concave], [axis_start, axis_end])[0].min() < 0.5:
		concave = None
	
	if (convex is None) or cdist([convex], [axis_start, axis_end])[0].min() < 0.5:
		convex = None
	
	# find intersection of axis with profile (axis_start)
	for mark in [concave, convex, axis_end]:
		if mark is None:
			continue
		norm = np.sqrt(((axis_start - mark)**2).sum())
		if norm > 0:
			vector = (axis_start - mark) / norm
			break
	d = np.sqrt(((axis_start - get_rim(profile))**2).sum())
	axis_start = profile[np.argmin(cdist((axis_start + vector * np.arange(0, d * 4, 0.1)[:,None]), profile).min(axis = 0))]
	
	return axis_start, concave, convex, axis_end

def get_landmark_vecors(axis_start, concave, convex, axis_end, params):
	
	neck_vector = None
	neck_length = None
	upper_vector = None
	upper_length = None
	lower_vector = None
	lower_length = None
	break_vector = None
	
	if concave is not None:
		neck_length = np.sqrt(((concave - axis_start)**2).sum())
		neck_vector = (concave - axis_start) / neck_length
	
	if convex is not None:
		vector1 = concave if (concave is not None) else axis_start
		
		upper_length = np.sqrt(((convex - vector1)**2).sum())
		upper_vector = (convex - vector1) / upper_length
		
		lower_length = np.sqrt(((axis_end - convex)**2).sum())
		lower_vector = (axis_end - convex) / lower_length
	
	if params:
		_, _, xd, yd = params
		break_vector = np.array([xd, yd])
	
	if (concave is None) and (convex is None):
		if axis_start[0] > axis_end[0]:
			neck_length = np.sqrt(((axis_end - axis_start)**2).sum())
			neck_vector = (axis_end - axis_start) / neck_length
		else:
			upper_length = np.sqrt(((axis_end - axis_start)**2).sum())
			upper_vector = (axis_end - axis_start) / upper_length
	
	if neck_vector is None:
		neck_length = 0
		neck_vector = upper_vector
	
	if upper_vector is None:
		if concave is not None:
			upper_length = np.sqrt(((axis_end - concave)**2).sum())
			upper_vector = (axis_end - concave) / upper_length
		else:
			upper_length = 0
			if break_vector is not None:
				upper_vector = break_vector
			else:
				upper_vector = neck_vector
	
	if lower_vector is None:
		lower_length = 0
		if break_vector is not None:
			lower_vector = break_vector
		else:
			lower_vector = neck_vector
	
	if break_vector is None:
		break_vector = lower_vector
	
	return [neck_vector, neck_length, upper_vector, upper_length, lower_vector, lower_length, break_vector]

def dist_worker(ijs_mp, collect_mp, profiles, sample_ids, arc_lengths, tangents, curvatures, landmarks):
	
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
		
		land1 = landmarks[sample_ids[i]] # [neck_vector1, neck_length1, upper_vector1, upper_length1, lower_vector1, lower_length1, break_vector1]
		land2 = landmarks[sample_ids[j]]
		
		shifts1 = profile1[(profile1[:,1] == profile1[:,1].min())][:,0]
		shifts2 = profile2[(profile2[:,1] == profile2[:,1].min())][:,0]
		
		h_dist = np.inf
		land_dist = np.inf
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
		land_dist = landmark_dist(land1, land2)
		
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
		distance[i, j, 5] = land_dist
		
		distance[j, i, 0] = R_dist
		distance[j, i, 1] = th_dist
		distance[j, i, 2] = kap_dist
		distance[j, i, 3] = h_dist
		distance[j, i, 4] = diam_dist
		distance[j, i, 5] = land_dist
		
	collect_mp.append(distance)
	
def calc_distances(profiles):
	# profiles[sample_id] = [profile, bottom, radius, thickness, params]
	
	profiles_n = len(profiles)
	sample_ids = list(profiles.keys())
	
	arc_lengths = {}
	tangents = {}
	curvatures = {}
	landmarks = {} # {sample_id: [upper_vector, upper_length, lower_vector, lower_length, break_vector], ...}
	for sample_id in profiles:
		profile, bottom, radius, thickness, params = profiles[sample_id]
		
		axis_start, concave, convex, axis_end = get_landmarks(profile, bottom, thickness, params)
		landmarks[sample_id] = get_landmark_vecors(axis_start, concave, convex, axis_end, params)
		
		prof = profile + [radius, 0]
		arc_lengths[sample_id] = arc_length(prof)
		tangents[sample_id] = tangent(prof)
		curvatures[sample_id] = np.gradient(tangents[sample_id])
		
	manager = mp.Manager()
	ijs_mp = manager.list(list(combinations(range(profiles_n), 2)))
	collect_mp = manager.list()
	
	procs = []
	for pi in range(mp.cpu_count()):
		procs.append(mp.Process(target = dist_worker, args = (ijs_mp, collect_mp, profiles, sample_ids, arc_lengths, tangents, curvatures, landmarks)))
		procs[-1].start()
	for proc in procs:
		proc.join()
	for proc in procs:
		proc.terminate()
		proc = None
	
	distance = np.ones((profiles_n, profiles_n, 6), dtype = float) # distance[i, j] = [R_dist, th_dist, kap_dist, h_dist, diam_dist, land_dist], ; where i, j = indices in sample_ids; R_dist, th_dist, kap_dist, h_dist, diam_dist, land_dist = components of morphometric distance
	for i in range(profiles_n):
		distance[i,i,:] = 0
	
	for dist in collect_mp:
		mask = (dist != -2)
		distance[mask] = dist[mask]
	
	return distance

def combine_dists(distance, w_R, w_th, w_kap, w_h, w_diam, w_land):
	# distance[i, j] = [R_dist, th_dist, kap_dist, h_dist, diam_dist, land_dist]; where i, j = indices in sample_ids; R_dist, th_dist, kap_dist, h_dist, diam_dist, land_dist = components of morphometric distance
	
	combined = np.ones(distance.shape[:2])
	
	w_sum = sum([w_R, w_th, w_kap, w_h, w_diam, w_land])
	w_R, w_th, w_kap, w_h, w_diam, w_land = [w / w_sum for w in [w_R, w_th, w_kap, w_h, w_diam, w_land]]
	
	dists = [None] * 6
	for idx in range(6):
		dists[idx] = distance[:,:,idx]
	R_dist, th_dist, kap_dist, h_dist, diam_dist, land_dist = dists
	
	land_dist[land_dist == -1] = land_dist[land_dist > -1].mean()
	
	combined = R_dist * w_R + th_dist * w_th + kap_dist * w_kap + h_dist * w_h + diam_dist * w_diam + land_dist * w_land
	
	for i in range(combined.shape[0]):
		combined[i,i] = 0
	
	return combined

def get_distmax_ordering(distance):
	# orders samples so that the the next one has the highest possible distance to the most similar of the previous samples
	# (ensures that the samples at the beginning of the sequence are the most diverse)
	# returns {sample_idx: order, ...}
	
	D = squareform(pdist(distance))
	
	# start with the profile most disstant from the centroid of the assemblage
	idx0 = np.argmax(((distance - distance.mean(axis = 0))**2).sum(axis = 1))
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
			sample_labels[idx] += ".%d" % (int(z_idx) + 1)
	for idx in sample_labels:
		sample_labels[idx] = sample_labels[idx][1:]
	
	return sample_labels

