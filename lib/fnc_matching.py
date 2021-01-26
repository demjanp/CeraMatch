import numpy as np
import multiprocessing as mp
from PIL import Image, ImageDraw
from itertools import combinations
from collections import defaultdict
from sklearn.decomposition import PCA
from skimage.morphology import skeletonize, dilation, square
from skimage import measure
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.cluster.hierarchy import linkage
from scipy.interpolate import interp1d
from scipy.signal import convolve2d
import networkx as nx

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

def rotate_coords(coords, angle):
	
	return np.dot(coords,np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]]))

'''
def __smoothen_coords(coords, step = 0.5, resolution = 0.5):
	# coords = [[x, y], ...]
	
	if step <= 0:
		return coords
	
	d = np.append(0, np.cumsum(np.sqrt(np.sum(np.diff(coords, axis = 0)**2, axis = 1))))
	coords = np.array([coords[np.abs(d - d[idx]) < step].mean(axis = 0) for idx in range(coords.shape[0])])
	
	collect = [coords[0]]
	for xy in coords[1:]:
		if not (xy == collect[-1]).all():
			collect.append(xy)
	coords = np.array(collect)
	
	d = np.append(0, np.cumsum(np.sqrt(np.sum(np.diff(coords, axis = 0)**2, axis = 1))))
	l = max(2, int(round(d[-1] / resolution)))
	alpha = np.linspace(0, d[-1], l)
	try:
		coords = interp1d(d, coords, kind = "cubic", axis = 0)(alpha)
	except:
		return np.array(collect)
	
	return coords
'''

def coords_between(p0, p1, step):
	
	x0, y0 = p0.astype(float)
	x1, y1 = p1.astype(float)
	l = int(round(max(abs(x1 - x0), abs(y1 - y0)) / step))
	if not l:
		return np.array([p0, p1])
	coords = np.vstack((np.linspace(x0, x1, l), np.linspace(y0, y1, l))).T
	if not (coords[0] == p0).all():
		coords = np.vstack((p0, coords))
	if not (coords[-1] == p1).all():
		coords = np.vstack((coords, p1))
	return coords

def get_interpolated(coords, step):
	
	d = np.sqrt(np.sum(np.diff(coords, axis = 0)**2, axis = 1))
	idxs_skips = np.where(d > step)[0]
	if idxs_skips.size:
		idxs_skips += 1
		filled = []
		for i, idx in enumerate(idxs_skips):
			filled.append(np.array(list(coords_between(coords[idx - 1], coords[idx], step))))
			if i < len(idxs_skips) - 1:
				idx0 = idx + 1
				idx1 = idxs_skips[i + 1] - 1
				if idx1 > idx0:
					filled.append(coords[idx0:idx1])
		if idxs_skips[0] > 1:
			filled = [coords[:idxs_skips[0] - 1]] + filled
		if idxs_skips[-1] < coords.shape[0] - 1:
			filled.append(coords[idxs_skips[-1] + 1:])
		if filled:
			coords = np.vstack(filled)
	return coords

def smoothen_coords(coords, step = 0.5, resolution = 0.5, closed = False):
	# coords = [[x, y], ...]
	
	if step <= 0.1:
		return coords
	
	coords = get_interpolated(coords, step)
	
	d = np.append(0, np.cumsum(np.sqrt(np.sum(np.diff(coords, axis = 0)**2, axis = 1))))
	coords = np.array([coords[np.abs(d - d[idx]) < step].mean(axis = 0) for idx in range(coords.shape[0])])
	
	collect = [coords[0]]
	for xy in coords[1:]:
		if not (xy == collect[-1]).all():
			collect.append(xy)
	coords = np.array(collect)
	
	if closed and (coords.shape[0] > 2):
		coords = np.vstack((coords[-2:], coords, coords[:3]))
	
	d = np.append(0, np.cumsum(np.sqrt(np.sum(np.diff(coords, axis = 0)**2, axis = 1))))
	l = max(2, int(round(d[-1] / resolution)))
	alpha = np.linspace(0, d[-1], l)
	try:
		coords = interp1d(d, coords, kind = "cubic", axis = 0)(alpha)
	except:
		return np.array(collect)
	
	if closed:
		l = max(1, coords.shape[0] // 3)
		idx = np.where((((coords[-l:][:,None] - coords[:l][None,:])**2).sum(axis = 2) < (resolution**2)).any(axis = 0))[0]
		if idx.size:
			idx = idx.max()
			coords = coords[idx:]
	
	return coords

def clean_mask(mask):
	
	labels = measure.label(mask, background = False)
	values = np.unique(labels)
	values = values[values > 0]
	value = sorted(values.tolist(), key = lambda value: (labels == value).sum())[-1]
	return (labels == value)

def profile_axis(profile, rasterize_factor, min_length = 10):
	
	[x0, y0], [x1, y1] = profile.min(axis = 0), profile.max(axis = 0)
	x0, y0, x1, y1 = int(x0) - 1, int(y0) - 1, int(x1) + 1, int(y1) + 1
	w, h = x1 - x0, y1 - y0
	
	img = Image.new("1", (w * rasterize_factor, h * rasterize_factor))
	draw = ImageDraw.Draw(img)
	profile = ((profile - [x0, y0]) * rasterize_factor).astype(int)
	draw.polygon([(x, y) for x, y in profile], fill = 0, outline = 1)
	outline = np.argwhere(np.array(img, dtype = bool).T)
	draw.polygon([(x, y) for x, y in profile], fill = 1, outline = 0)
	profile_mask = clean_mask(np.array(img, dtype = bool).T)
	img.close()
	
	skelet = skeletonize(profile_mask)
	connectivity = convolve2d(skelet, square(3), mode = "same", boundary = "fill", fillvalue = False)
	nodes = (skelet & ((connectivity > 3) | (connectivity == 2)))
	
	rcs = np.argwhere(skelet)
	idxs_nodes = set(np.where(nodes[rcs[:,0], rcs[:,1]])[0].tolist())
	d = np.abs(np.dstack((rcs[:,0,None] - rcs[None,:,0], rcs[:,1,None] - rcs[None,:,1]))).max(axis = 2)
	G = nx.from_numpy_matrix(d == 1)
	paths = []
	for i, j in combinations(idxs_nodes, 2):
		path = nx.shortest_path(G, i, j)
		if len(idxs_nodes.intersection(path)) > 2:
			continue
		paths.append(rcs[path])
	paths = sorted(paths, key = lambda path: len(path))
	
	axis = np.array(paths.pop(), dtype = float)	
	
	if ((axis[-1] / rasterize_factor - [-x0, -y0])**2).sum() < ((axis[0] / rasterize_factor - [-x0, -y0])**2).sum():
		axis = axis[::-1]
	
	d_min = cdist(axis, profile).min(axis = 1)
	d_min = d_min[d_min > d_min.max() / 2].mean() / 2
	
	d_max = rasterize_factor * 2
	while paths:
		path = paths.pop()
		
		if cdist(path, profile).min() < d_min:
			break
		
		d1 = np.sqrt(((path[0] - axis[0])**2).sum())
		d2 = np.sqrt(((path[0] - axis[-1])**2).sum())
		d3 = np.sqrt(((path[-1] - axis[0])**2).sum())
		d4 = np.sqrt(((path[-1] - axis[-1])**2).sum())
		
		if (d1 < d_max) and (d3 > d_max):
			axis = np.vstack((path[::-1], axis))
		elif (d2 < d_max) and (d4 > d_max):
			axis = np.vstack((axis, path))
		elif (d3 < d_max) and (d1 > d_max):
			axis = np.vstack((path, axis))
		elif (d4 < d_max) and (d2 > d_max):
			axis = np.vstack((axis, path[::-1]))
	
	d_min = cdist(axis, profile).min(axis = 1)
	d_min = d_min[d_min > d_min.max() / 2].mean()
	
	axis0 = axis.copy()
	d = cdist([profile[np.argmin(cdist([axis[-1]], profile)[0])]], axis)[0]
	axis = axis[d > d_min]
	if axis.shape[0] < min_length:
		axis = axis0.copy()
	
	axis0 = axis.copy()
	d = cdist([profile[np.argmin(cdist([axis[0]], profile)[0])]], axis)[0]
	axis = axis[d > d_min]
	if axis.shape[0] < min_length:
		axis = axis0.copy()
	
	thickness = cdist(axis, profile).min(axis = 1) * 2
	thickness = thickness[thickness > thickness.max() / 2].mean() / rasterize_factor
	
	axis = axis / rasterize_factor + [x0, y0]
	
	axis = smoothen_coords(axis)
	
	return axis, thickness

def diameter_dist(axis1, radius1, axis2, radius2):
	
	axis1 = axis1 + [radius1, 0]
	axis2 = axis2 + [radius2, 0]
	ymax = min(axis1[:,1].max(), axis2[:,1].max())
	axis1 = axis1[axis1[:,1] <= ymax]
	axis2 = axis2[axis2[:,1] <= ymax]
	if (not axis1.size) or (not axis2.size):
		print("diam dist error")
		return 1
	d_sum = 0
	d_norm = 0
	for i in range(axis1.shape[0]):
		j = np.argmin(np.abs(axis2[:,1] - axis1[i,1]))
		r1 = axis1[i,0]
		r2 = axis2[j,0]
		if r1 > r2:
			r1, r2 = r2, r1
		d_sum += r1
		d_norm += r2
	for i in range(axis2.shape[0]):
		j = np.argmin(np.abs(axis1[:,1] - axis2[i,1]))
		r1 = axis2[i,0]
		r2 = axis1[j,0]
		if r1 > r2:
			r1, r2 = r2, r1
		d_sum += r1
		d_norm += r2
	return 1 - d_sum / d_norm

def axis_dist(axis1, axis2):
	
	idx_max = min(axis1.shape[0], axis2.shape[0])
	axis1 = axis1[:idx_max] - axis1[0]
	axis2 = axis2[:idx_max] - axis2[0]
	return (((axis1 - axis2)**2).sum(axis = 1)**0.5).sum() / (np.arange(idx_max)*2).sum()

def find_keypoints(profile, axis, thickness):
	
	profile = smoothen_coords(profile, closed = False)
	
	step = thickness
	curv_profile = np.zeros(profile.shape[0], dtype = float)
	profile_pos = np.hstack(([0], np.cumsum((np.diff(profile, axis = 0) ** 2).sum(axis = 1))))
	for i in range(profile.shape[0]):
		segment = profile[np.abs(profile_pos - profile_pos[i]) < thickness]
		if segment.shape[0] < 3:
			continue
		angle = np.arctan2(*(segment[-1] - segment[0])[::-1])
		if angle > 0:
			angle = angle + np.pi
		angle += np.pi / 2
		segment = rotate_coords(segment, -angle).astype(float)
		segment = segment[:,0] - segment[:,0].min()
		segment_inv = -segment
		segment_inv -= segment_inv.min()
		curv_profile[i] = max(np.trapz(segment), np.trapz(segment_inv))
	
	idx_rim = np.argmin(cdist([[0,0]], profile)[0])
	d_axis_profile = cdist(axis, profile)
	curvature = np.zeros(axis.shape[0], dtype = float)
	for i in range(axis.shape[0]):
		if idx_rim == 0:
			curvature[i] = curv_profile[idx_rim:][np.argmin(d_axis_profile[i][idx_rim:])]
		elif idx_rim == curv_profile.shape[0] - 1:
			curvature[i] = curv_profile[:idx_rim][np.argmin(d_axis_profile[i][:idx_rim])]
		else:
			curvature[i] = max(curv_profile[:idx_rim][np.argmin(d_axis_profile[i][:idx_rim])], curv_profile[idx_rim:][np.argmin(d_axis_profile[i][idx_rim:])])
	
	axis_pos = np.hstack(([0], np.cumsum((np.diff(axis, axis = 0) ** 2).sum(axis = 1))))
	keypoints = np.zeros(axis.shape[0], dtype = bool)
	keypoints[0] = True
	vals = np.sort(np.unique(curvature))[::-1]
	labels = measure.label(curvature)
	for val in vals:
		labels_val = labels[curvature == val]
		for label in np.unique(labels_val):
			idx = int(round(np.where(labels == label)[0].mean()))
			mask = (np.abs(axis_pos - axis_pos[idx]) < thickness / 2)
			if not keypoints[mask].any():
				keypoints[idx] = True
	
	return axis[keypoints]

def frame_hamming_scaled(profile1, keypoints1, profile2, keypoints2, r, rasterize_factor):
	
	def _make_mask(profile, keypoints, angle, scale, r_r, rasterize_factor):
		
		if angle != 0:
			profile = rotate_coords(profile, angle)
			keypoints = rotate_coords(keypoints, angle)
		profile = np.round(profile * scale * rasterize_factor).astype(int)
		keypoints = np.round(keypoints * scale * rasterize_factor).astype(int)
		x0, y0 = profile.min(axis = 0) - 4*r_r
		profile -= [x0, y0]
		keypoints -= [x0, y0]
		w, h = profile.max(axis = 0) + 4*r_r
		img = Image.new("1", (w, h))
		draw = ImageDraw.Draw(img)
		draw.polygon([(x, y) for x, y in profile], fill = 1)
		mask = np.array(img, dtype = bool).T
		img.close()
		return mask, keypoints
	
	def _get_kp_mask(mask, keypoints, i, shift_x, shift_y, r_r, rasterize_factor):
		
		row, col = np.round(keypoints[i] - [shift_x * rasterize_factor, shift_y * rasterize_factor]).astype(int)
		col0 = col - r_r
		col1 = col0 + 2*r_r + 1
		row0 = row - r_r
		row1 = row0 + 2*r_r + 1
		return mask[:,col0:col1][row0:row1]
	
	r = int(round(r))
	r_r = r * rasterize_factor
	w = np.ones((2*r_r + 1, 2*r_r + 1), dtype = float)
	ijs = np.argwhere(w > 0)
	d = np.sqrt(((ijs - r_r)**2).sum(axis = 1))
	d[d > r_r] = r_r
	w[ijs[:,0], ijs[:,1]] = ((r_r - d) / r_r)**2
	
	rot_step = 2*np.arcsin(1 / (2*r))
	angle_min = -np.pi / 16
	angle_max = np.pi / 16
	angles = np.linspace(angle_min, angle_max, int(round((angle_max - angle_min) / rot_step)))
	angles = angles[angles != 0]
	angles = np.insert(angles, 0, 0)
	
	scales = np.linspace(0.8, 1.2, 5)
	scales = scales[scales != 1]
	scales = np.insert(scales, 0, 1)
	
	shifts = []
	for shift_x in range(-4, 5, 2):
		for shift_y in range(-4, 5, 2):
			if [shift_x, shift_y] == [0, 0]:
				continue
			shifts.append([shift_x, shift_y])
	shifts = [[0,0]] + shifts
	
	h_dist_sum = 0
	h_dist_norm = 0
	d = cdist(keypoints1, keypoints2)
	mask_m1, keypoints_m1 = _make_mask(profile1, keypoints1, 0, 1, r_r, rasterize_factor)
	for i in range(keypoints1.shape[0]):
		jj = np.where(d[i] < r)[0]
		if not jj.size:
			continue
		mask1 = _get_kp_mask(mask_m1, keypoints_m1, i, 0, 0, r_r, rasterize_factor)
		mask1_sum = w[mask1].sum()
		h_dist_opt = np.inf
		for angle in angles:
			for scale in scales:
				mask_m2, keypoints_m2 = _make_mask(profile2, keypoints2, angle, scale, r_r, rasterize_factor)
				for j in jj:
					for shift_x, shift_y in shifts:
						mask2 = _get_kp_mask(mask_m2, keypoints_m2, j, shift_x, shift_y, r_r, rasterize_factor)
						h_dist = 1 - (2*w[mask1 & mask2].sum()) / (mask1_sum + w[mask2].sum())
						if h_dist < h_dist_opt:
							h_dist_opt = h_dist
		
		if h_dist_opt < np.inf:
			h_dist_sum += h_dist_opt**2
			h_dist_norm += 1
	
	return h_dist_sum, h_dist_norm

def frame_hamming(profile1, keypoints1, profile2, keypoints2, r, rasterize_factor):
	
	def _make_mask(profile, keypoints, angle, r_r, rasterize_factor):
		
		if angle != 0:
			profile = rotate_coords(profile, angle)
			keypoints = rotate_coords(keypoints, angle)
		profile = np.round(profile * rasterize_factor).astype(int)
		keypoints = np.round(keypoints * rasterize_factor).astype(int)
		x0, y0 = profile.min(axis = 0) - 4*r_r
		profile -= [x0, y0]
		keypoints -= [x0, y0]
		w, h = profile.max(axis = 0) + 4*r_r
		img = Image.new("1", (w, h))
		draw = ImageDraw.Draw(img)
		draw.polygon([(x, y) for x, y in profile], fill = 1)
		mask = np.array(img, dtype = bool).T
		img.close()
		return mask, keypoints
	
	def _get_kp_mask(mask, keypoints, i, shift_x, shift_y, r_r, rasterize_factor):
		
		row, col = np.round(keypoints[i] - [shift_x * rasterize_factor, shift_y * rasterize_factor]).astype(int)
		col0 = col - r_r
		col1 = col0 + 2*r_r + 1
		row0 = row - r_r
		row1 = row0 + 2*r_r + 1
		return mask[:,col0:col1][row0:row1]
	
	r = int(round(r))
	r_r = r * rasterize_factor
	w = np.ones((2*r_r + 1, 2*r_r + 1), dtype = float)
	ijs = np.argwhere(w > 0)
	d = np.sqrt(((ijs - r_r)**2).sum(axis = 1))
	d[d > r_r] = r_r
	w[ijs[:,0], ijs[:,1]] = ((r_r - d) / r_r)**2
	
	rot_step = 2*np.arcsin(1 / (2*r))
	angle_min = -np.pi / 8
	angle_max = np.pi / 8
	angles = np.linspace(angle_min, angle_max, int(round((angle_max - angle_min) / rot_step)))
	angles = angles[angles != 0]
	angles = np.insert(angles, 0, 0)
	
	shifts = []
	for shift_x in range(-4, 5, 2):
		for shift_y in range(-4, 5, 2):
			if [shift_x, shift_y] == [0, 0]:
				continue
			shifts.append([shift_x, shift_y])
	shifts = [[0,0]] + shifts
	
	h_dist_sum = 0
	h_dist_norm = 0
	d = cdist(keypoints1, keypoints2)
	mask_m1, keypoints_m1 = _make_mask(profile1, keypoints1, 0, r_r, rasterize_factor)
	for i in range(keypoints1.shape[0]):
		jj = np.where(d[i] < 2*r)[0]
		if not jj.size:
			continue
		mask1 = _get_kp_mask(mask_m1, keypoints_m1, i, 0, 0, r_r, rasterize_factor)
		mask1_sum = w[mask1].sum()
		h_dist_opt = np.inf
		for angle in angles:
			mask_m2, keypoints_m2 = _make_mask(profile2, keypoints2, angle, r_r, rasterize_factor)
			for j in jj:
				for shift_x, shift_y in shifts:
					mask2 = _get_kp_mask(mask_m2, keypoints_m2, j, shift_x, shift_y, r_r, rasterize_factor)
					h_dist = 1 - (2*w[mask1 & mask2].sum()) / (mask1_sum + w[mask2].sum())
					if h_dist < h_dist_opt:
						h_dist_opt = h_dist
		
		if h_dist_opt < np.inf:
			h_dist_sum += h_dist_opt**2
			h_dist_norm += 1
	
	return h_dist_sum, h_dist_norm

def dist_worker(ijs_mp, collect_mp, data, sample_ids, cmax):
	
	rasterize_factor = 4
	profiles_n = len(data)
	distance = np.zeros((profiles_n, profiles_n, 4), dtype = float) - 2
	while True:
		try:
			i, j = ijs_mp.pop()
		except:
			break
		cnt = cmax - len(ijs_mp)
		print("\rprocessing {:d}/{:d} ({:%})          ".format(cnt, cmax, cnt / cmax), end = "")
		
		profile1, axis1, radius1, keypoints1, thickness1 = data[sample_ids[i]]
		profile2, axis2, radius2, keypoints2, thickness2 = data[sample_ids[j]]
		
		r = 2*max(thickness1, thickness2)
		
		diam_dist = diameter_dist(axis1, radius1, axis2, radius2)
		ax_dist = axis_dist(axis1, axis2)
		
		h_dist_sum1, h_dist_norm1 = frame_hamming(profile1, keypoints1, profile2, keypoints2, r, rasterize_factor)
		h_dist_sum2, h_dist_norm2 = frame_hamming(profile2, keypoints2, profile1, keypoints1, r, rasterize_factor)
		h_dist = np.sqrt((h_dist_sum1 + h_dist_sum2) / (h_dist_norm1 + h_dist_norm2))
		
		keypoints1 = np.array([keypoints1[0]])
		keypoints2 = np.array([keypoints2[0]])
		h_dist_sum1, h_dist_norm1 = frame_hamming(profile1, keypoints1, profile2, keypoints2, r, rasterize_factor)
		h_dist_sum2, h_dist_norm2 = frame_hamming(profile2, keypoints2, profile1, keypoints1, r, rasterize_factor)
		h_rim_dist = np.sqrt((h_dist_sum1 + h_dist_sum2) / (h_dist_norm1 + h_dist_norm2))
		
		distance[i, j, 0] = diam_dist
		distance[i, j, 1] = ax_dist
		distance[i, j, 2] = h_dist
		distance[i, j, 3] = h_rim_dist
		
		distance[j, i, 0] = diam_dist
		distance[j, i, 1] = ax_dist
		distance[j, i, 2] = h_dist
		distance[j, i, 3] = h_rim_dist
		
	collect_mp.append(distance)
	
def calc_distances(profiles, distance = None):
	# profiles[sample_id] = [profile, radius]
	# returns distance[i, j] = [diam_dist, ax_dist, h_dist, h_rim_dist], ; where i, j = indices in sample_ids; [name]_dist = components of morphometric distance
	
	rasterize_factor = 10
	
	profiles_n = len(profiles)
	sample_ids = list(profiles.keys())
	
	manager = mp.Manager()
	if distance is None:
		distance = np.ones((profiles_n, profiles_n, 4), dtype = float)
		ijs_mp = manager.list(list(combinations(range(profiles_n), 2)))
	else:
		ijs = []
		for i, j in combinations(range(profiles_n), 2):
			if (distance[i,j] == np.inf).any():
				ijs.append([i,j])
		if not ijs:
			return distance
		ijs_mp = manager.list(ijs)
	
	data = {} # {sample_id: [profile, axis, radius, keypoints, thickness], ...}
	cmax = len(profiles)
	cnt = 1
	for sample_id in profiles:
		print("\rpreparing data {:0.0%}".format(cnt / cmax), end = "")
		cnt += 1
		profile, radius = profiles[sample_id]
		axis, thickness = profile_axis(profile, rasterize_factor)
		if axis.shape[0] < 10:
			print("\naxis error: %s (%d)\n" % (sample_id, axis.shape[0])) # DEBUG
			raise
		axis = axis - axis[0]
		keypoints = find_keypoints(profile, axis, thickness)
		data[sample_id] = [profile, axis, radius, keypoints, thickness]
	collect_mp = manager.list()
	cmax = len(ijs_mp)
	procs = []
#	for pi in range(mp.cpu_count()):
	for pi in range(min(30, mp.cpu_count())):
		procs.append(mp.Process(target = dist_worker, args = (ijs_mp, collect_mp, data, sample_ids, cmax)))
		procs[-1].start()
	for proc in procs:
		proc.join()
	for proc in procs:
		proc.terminate()
		proc = None
	
	# distance[i, j] = [diam_dist, ax_dist, h_dist, h_rim_dist], ; where i, j = indices in sample_ids; [name]_dist = components of morphometric distance
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

def get_distmax_ordering(distance):
	# orders samples so that the the next one has the highest possible distance to the most similar of the previous samples
	# (ensures that the samples at the beginning of the sequence are the most diverse)
	# returns {sample_idx: order, ...}
	
	# start with the profile most disstant from the centroid of the assemblage
	idx0 = sorted(np.unique(np.argwhere(distance == distance.max())).tolist(), key = lambda i: distance[i].mean())[-1]
	idxs_done = [idx0]
	data = {} # {sample_idx: order, ...}
	i = 0
	data[idx0] = i
	idxs_todo = [idx for idx in range(distance.shape[0]) if idx != idx0]
	while idxs_todo:
		i += 1
		d = distance[:,idxs_todo][idxs_done]
		d_todo = d.max(axis = 0)
		idx1 = idxs_todo[sorted(np.where(d_todo == d_todo.max())[0].tolist(), key = lambda idx: d[:,idx].mean())[-1]]
		data[idx1] = i
		idxs_done.append(idx1)
		idxs_todo.remove(idx1)
	return data

def get_hca_labels(distance):
	# returns [label, ...]
	
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
	sample_labels = [sample_labels[idx] for idx in range(samples_n)]
	
	return sample_labels

def get_labels(clusters, D):
	# hierarchically label clusters

	def _calc_dist(cluster, D):
		
		if len(cluster) < 2:
			return 0
		return np.nanmean(D[:,cluster][cluster])
	
	D[np.diag_indices(D.shape[0])] = 0
	mask = (D == 0)
	mins = []
	for idx in range(D.shape[2]):
		d = D[:,:,idx]
		mins.append(np.nanmin(d[d > 0]))
	D = D - mins
	D[mask] = 0
	D[D == np.inf] = np.nan
	D = D / np.nanmean(D, axis = (0,1))
	
	D = np.sqrt(np.nansum(D**2, axis = 2))
	D[np.diag_indices(D.shape[0])] = 0
	
	clusters2 = {}
	collect = {}
	for ci, label in enumerate(list(clusters.keys())):
		clusters2[ci] = clusters[label].copy()
		collect[ci] = clusters[label].copy()
	clusters = collect
	
	d_comb = np.zeros((len(clusters2), len(clusters2)), dtype = float) + np.inf
	for c1, c2 in combinations(list(clusters2.keys()), 2):
		d_comb[c1,c2] = _calc_dist(clusters2[c1] + clusters2[c2], D)
		d_comb[c2,c1] = d_comb[c1,c2]
	
	joins = []
	cmax = np.prod(d_comb.shape)
	cnt_last = 0
	while True:
		c1, c2 = np.argwhere(d_comb == d_comb.min())[0]
		if d_comb[c1,c2] == np.inf:
			break
		cnt = (d_comb == np.inf).sum()
		print("\rlabeling {:0.0%} {:d}         ".format(cnt / cmax, len(clusters2)), end = "")
		d_comb[c1,c2] = np.inf
		d_comb[c2,c1] = np.inf
		clusters2[c1] = clusters2[c1] + clusters2[c2]
		joins.append([c1, c2])
		del clusters2[c2]
		d_comb[c2] = np.inf
		d_comb[:,c2] = np.inf
		for c2 in clusters2:
			if (c2 == c1):
				continue
			d_comb[c1,c2] = _calc_dist(clusters2[c1] + clusters2[c2], D)
			d_comb[c2,c1] = d_comb[c1,c2]
	
	labels_dict = {}
	label = len(joins) + 1
	for cis in joins:
		for ci in cis:
			if ci not in labels_dict:
				labels_dict[ci] = [str(label)]
				label += 1
	for ci in clusters:
		if ci not in labels_dict:
			labels_dict[ci] = [str(label)]
			label += 1
	z = 1
	for c1, c2 in joins[::-1]:
		label = str(z)
		labels_dict[c1] = [label] + labels_dict[c1]
		labels_dict[c2] = labels_dict[c1][:-1] + labels_dict[c2]
		z += 1
	for ci in clusters:
		labels_dict[ci] = ".".join(labels_dict[ci][:-2][::-1] + [labels_dict[ci][-1]])
		
	clusters = dict([[labels_dict[ci], clusters[ci]] for ci in clusters])
	
	return clusters

def get_clusters(D, limit = 0.68):
	# D[i, j] = [diam_dist, ax_dist, h_dist], ; where i, j = indices in sample_ids; [name]_dist = components of morphometric distance
	# returns clusters = {label: [i, ...], ...}; i = index in sample_ids
	
	def _calc_dist(cluster, D, limit = None):
		
		d = D[:,cluster][cluster]
		if (limit is not None) and (np.nanmax(d) > limit):
			return np.inf
		return np.nanmean(d)
	
	D[np.diag_indices(D.shape[0])] = 0
	mask = (D == 0)
	mins = []
	for idx in range(D.shape[2]):
		d = D[:,:,idx]
		mins.append(np.nanmin(d[d > 0]))
	D = D - mins
	D[mask] = 0
	D[D == np.inf] = np.nan
	D = D / np.nanmean(D, axis = (0,1))
	
	D_num = np.sqrt(np.nansum(D**2, axis = 2))
	D_num[np.diag_indices(D_num.shape[0])] = 0
	D_num[np.isnan(D_num)] = 2*np.nanmax(D_num)
	
	limit = np.nanmean(D) * limit
	
	clusters = dict([[i, [i]] for i in range(D.shape[0])])
	
	d_comb = np.zeros((len(clusters), len(clusters)), dtype = float) + np.inf
	for c1, c2 in combinations(list(clusters.keys()), 2):
		d_comb[c1,c2] = _calc_dist(clusters[c1] + clusters[c2], D, limit)
		d_comb[c2,c1] = d_comb[c1,c2]
	
	cmax = np.prod(d_comb.shape)
	cnt_last = 0
	
	while True:
		c1, c2 = np.argwhere(d_comb == d_comb.min())[0]
		if d_comb[c1,c2] == np.inf:
			break
		
		cnt = (d_comb == np.inf).sum()
		if cnt - cnt_last > 10:
			print("\rclustering {:0.0%} {:d}       ".format(cnt / cmax, len(clusters)), end = "")
			cnt_last = cnt
		
		d_comb[c1,c2] = np.inf
		d_comb[c2,c1] = np.inf
		clusters[c1] = clusters[c1] + clusters[c2]
		del clusters[c2]
		d_comb[c2] = np.inf
		d_comb[:,c2] = np.inf
		for c2 in clusters:
			if (c2 == c1):
				continue
			d_comb[c1,c2] = _calc_dist(clusters[c1] + clusters[c2], D, limit)
			d_comb[c2,c1] = d_comb[c1,c2]
	
	clusters = dict([(str(ci), clusters[label]) for ci, label in enumerate(list(clusters.keys()))])
	
	for label in clusters:
		if len(clusters[label]) < 2:
			continue
		scores = calc_pca_scores(D_num[:,clusters[label]][clusters[label]])
		center = scores.mean(axis = 0)
		clusters[label] = [clusters[label][idx] for idx in np.argsort(cdist([center], scores)[0])]
	
	return clusters
	
def get_clusters_2(D, type_idxs = None):
	# D[i, j] = [diam_dist, ax_dist, h_dist], ; where i, j = indices in sample_ids; [name]_dist = components of morphometric distance
	# returns clusters = {label: [i, ...], ...}; i = index in sample_ids
	
	D[np.diag_indices(D.shape[0])] = 0
	mask = (D == 0)
	mins = []
	for idx in range(D.shape[2]):
		d = D[:,:,idx]
		mins.append(np.nanmin(d[d > 0]))
	D = D - mins
	D[mask] = 0
	D[D == np.inf] = np.nan
	D = D / np.nanmean(D, axis = (0,1))
	
	D_nan = np.sqrt(np.nansum(D**2, axis = 2))
	D_nan[np.diag_indices(D_nan.shape[0])] = 0
	
	D_inf = D_nan.copy()
	D_inf[np.diag_indices(D_inf.shape[0])] = np.inf
	D_inf[D_inf == 0] = np.inf
	
	D_num = D_nan.copy()
	D_num[np.isnan(D_num)] = 2*np.nanmax(D_num)
	
	if type_idxs is None:
		G = nx.Graph()
		for i in range(D_inf.shape[0]):
			j = np.argmin(D_inf[i])
			G.add_edge(i, j)
		
		clusters = [list(G.subgraph(c).nodes) for c in nx.connected_components(G)]
		
		type_idxs = set([])
		for cluster in clusters:
			scores = calc_pca_scores(D_num[:,cluster][cluster])
			center = scores.mean(axis = 0)
			type_idxs.add(cluster[np.argmin(cdist([center], scores)[0])])
		type_idxs = list(type_idxs)
	
	collect = dict([(int(i), [int(i)]) for i in type_idxs])
	ii = sorted(list(set(range(D_inf.shape[0])).difference(type_idxs)))
	for i in ii:
		j = np.argmin(D_inf[i, type_idxs])
		collect[int(type_idxs[j])].append(int(i))
	clusters = {}
	for i in type_idxs:
		clusters[str(i)] = sorted(collect[i], key = lambda j: D_inf[i,j])
	
	return clusters


def get_clusters_3(D):
	# D[i, j] = [diam_dist, ax_dist, h_dist], ; where i, j = indices in sample_ids; [name]_dist = components of morphometric distance
	# returns clusters = {label: [i, ...], ...}; i = index in sample_ids
	
	D[np.diag_indices(D.shape[0])] = 0
	mask = (D == 0)
	mins = []
	for idx in range(D.shape[2]):
		d = D[:,:,idx]
		mins.append(np.nanmin(d[d > 0]))
	D = D - mins
	D[mask] = 0
	D[D == np.inf] = np.nan
	D = D / np.nanmean(D, axis = (0,1))
	
	D_nan = np.sqrt(np.nansum(D**2, axis = 2))
	D_nan[np.diag_indices(D_nan.shape[0])] = 0
	
	D_inf = D_nan.copy()
	D_inf[np.diag_indices(D_inf.shape[0])] = np.inf
	D_inf[D_inf == 0] = np.inf
	
	D_num = D_nan.copy()
	D_num[np.isnan(D_num)] = 2*np.nanmax(D_num)

	G = nx.Graph()
	for i in range(D_inf.shape[0]):
		j = np.argmin(D_inf[i])
		G.add_edge(i, j)
	clusters = dict([(str(ci), [int(i) for i in G.subgraph(c).nodes]) for ci, c in enumerate(nx.connected_components(G))])
	
	'''
	# this is only needed if some samples don't get a closest match in the previous step
	done = set(G.nodes)
	label = len(clusters)
	for i in range(D_inf.shape[0]):
		if i not in done:
			clusters[str(label)] = [i]
			label += 1
	'''
	
	for label in clusters:
		if len(clusters[label]) < 2:
			continue
		scores = calc_pca_scores(D_num[:,clusters[label]][clusters[label]])
		center = scores.mean(axis = 0)
		clusters[label] = [clusters[label][idx] for idx in np.argsort(cdist([center], scores)[0])]
	
	return clusters

