import numpy as np
import multiprocessing as mp
from PIL import Image, ImageDraw
from itertools import combinations
from collections import defaultdict
from natsort import natsorted
from sklearn.decomposition import PCA
from skimage.morphology import skeletonize, dilation, square
from skimage import measure
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.cluster.hierarchy import linkage
from scipy.interpolate import interp1d
from scipy.signal import convolve2d
import networkx as nx
import time

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

def rotate_coords(coords, angle):
	
	return np.dot(coords,np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]]))

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

def get_reduced(coords, step):
	
	d = np.hstack(([0], np.sqrt(np.sum(np.diff(coords, axis = 0)**2, axis = 1))))
	idxs = []
	d_cum = 0
	for i in range(coords.shape[0]):
		d_cum += d[i]
		if d_cum >= step:
			idxs.append(i)
			d_cum = 0
	if len(idxs) < 10:
		return coords
	if idxs[-1] != idxs[0]:
		idxs.append(idxs[0])
	return coords[idxs]

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

def dice_dist(profile1, keypoints1, profile2, keypoints2, r, rasterize_factor):
	
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
	
	d_dist_sum = 0
	d_dist_norm = 0
	d = cdist(keypoints1, keypoints2)
	mask_m1, keypoints_m1 = _make_mask(profile1, keypoints1, 0, r_r, rasterize_factor)
	for i in range(keypoints1.shape[0]):
		jj = np.where(d[i] < 2*r)[0]
		if not jj.size:
			continue
		mask1 = _get_kp_mask(mask_m1, keypoints_m1, i, 0, 0, r_r, rasterize_factor)
		mask1_sum = w[mask1].sum()
		d_dist_opt = np.inf
		for angle in angles:
			mask_m2, keypoints_m2 = _make_mask(profile2, keypoints2, angle, r_r, rasterize_factor)
			for j in jj:
				for shift_x, shift_y in shifts:
					mask2 = _get_kp_mask(mask_m2, keypoints_m2, j, shift_x, shift_y, r_r, rasterize_factor)
					d_dist = 1 - (2*w[mask1 & mask2].sum()) / (mask1_sum + w[mask2].sum())
					if d_dist < d_dist_opt:
						d_dist_opt = d_dist
		
		if d_dist_opt < np.inf:
			d_dist_sum += d_dist_opt**2
			d_dist_norm += 1
	
	return d_dist_sum, d_dist_norm

def dist_worker(ijs_mp, collect_mp, data, sample_ids, select_components):
	
	do_diam, do_axis, do_dice, do_dice_rim = select_components
	rasterize_factor = 4
	profiles_n = len(data)
	distance = np.zeros((profiles_n, profiles_n, 4), dtype = float) - 2
	while True:
		try:
			i, j = ijs_mp.pop()
		except:
			break
		
		profile1, axis1, radius1, keypoints1, thickness1 = data[sample_ids[i]]
		profile2, axis2, radius2, keypoints2, thickness2 = data[sample_ids[j]]
		
		if (profile1 is None) or (profile2 is None):
			continue
		
		r = 2*max(thickness1, thickness2)
		
		if do_diam:
			diam_dist = diameter_dist(axis1, radius1, axis2, radius2)
		else:
			diam_dist = 0.1
		if do_axis:
			ax_dist = axis_dist(axis1, axis2)
		else:
			ax_dist = 0.1
		
		if do_dice:
			d_dist_sum1, d_dist_norm1 = dice_dist(profile1, keypoints1, profile2, keypoints2, r, rasterize_factor)
			d_dist_sum2, d_dist_norm2 = dice_dist(profile2, keypoints2, profile1, keypoints1, r, rasterize_factor)
			d_dist = np.sqrt((d_dist_sum1 + d_dist_sum2) / (d_dist_norm1 + d_dist_norm2))
		else:
			d_dist = 0.1
		
		if do_dice_rim:
			keypoints1 = np.array([keypoints1[0]])
			keypoints2 = np.array([keypoints2[0]])
			d_dist_sum1, d_dist_norm1 = dice_dist(profile1, keypoints1, profile2, keypoints2, r, rasterize_factor)
			d_dist_sum2, d_dist_norm2 = dice_dist(profile2, keypoints2, profile1, keypoints1, r, rasterize_factor)
			d_rim_dist = np.sqrt((d_dist_sum1 + d_dist_sum2) / (d_dist_norm1 + d_dist_norm2))
		else:
			d_rim_dist = 0.1
		
		distance[i, j, 0] = diam_dist
		distance[i, j, 1] = ax_dist
		distance[i, j, 2] = d_dist
		distance[i, j, 3] = d_rim_dist
		
		distance[j, i, 0] = diam_dist
		distance[j, i, 1] = ax_dist
		distance[j, i, 2] = d_dist
		distance[j, i, 3] = d_rim_dist
		
	collect_mp.append(distance)
	
def calc_distances(profiles, select_components, distance = None, progress = None):
	# profiles[sample_id] = [profile, radius]
	# select_components = [diam, axis, dice, dice_rim]; True/False - calculate component
	# returns distance[i, j] = [diam_dist, ax_dist, d_dist, d_rim_dist], ; where i, j = indices in sample_ids; [name]_dist = components of morphometric distance
	
	def _update_progress(ijs_mp, procs):
		
		cnt = cmax - len(ijs_mp)
		if progress is None:
			print("\rclustering %d/%d            " % (cnt, cmax), end = "")
		else:
			progress.update_state(text = "Calculating Distances...", value = max(0, cnt - 1), maximum = cmax)
			if progress.cancel_pressed():
				for proc in procs:
					proc.terminate()
					proc = None
				return -1
		return cnt
	
	rasterize_factor = 10
	
	do_diam, do_axis, do_dice, do_dice_rim = select_components
	profiles_n = len(profiles)
	sample_ids = list(profiles.keys())
	
	cmax = profiles_n
	progress.update_state(text = "Analyzing Profiles...", value = 0, maximum = cmax)
	
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
	cnt = 1
	for sample_id in profiles:
		profile, radius = profiles[sample_id]
		axis, thickness = profile_axis(profile, rasterize_factor)
		if axis.shape[0] < 10:
			print("\naxis error: %s (%d)\n" % (sample_id, axis.shape[0])) # DEBUG
			data[sample_id] = [None, None, None, None, None]
			continue
		axis = axis - axis[0]
		if do_dice or do_dice_rim:
			keypoints = find_keypoints(profile, axis, thickness)
		else:
			keypoints = []
		data[sample_id] = [profile, axis, radius, keypoints, thickness]
		
		if progress is None:
			print("\rpreparing data %d/%d" % (cnt, cmax), end = "")
		else:
			progress.update_state(text = "Analyzing Profiles...", value = cnt, maximum = cmax)
			if progress.cancel_pressed():
				return None
		cnt += 1
	
	if progress is not None:
		progress.update_state(text = "Calculating Distances...", value = 0, maximum = cmax)
	collect_mp = manager.list()
	cmax = len(ijs_mp)
	procs = []
	for pi in range(min(30, mp.cpu_count() - 1)):
		procs.append(mp.Process(target = dist_worker, args = (ijs_mp, collect_mp, data, sample_ids, select_components)))
		procs[-1].start()
		cnt = _update_progress(ijs_mp, procs)
		if cnt == -1:
			return None
	
	progress.update_state(text = "Calculating Distances...", value = 0, maximum = cmax)
	
	cnt = 1
	while cnt < cmax:
		cnt = _update_progress(ijs_mp, procs)
		if cnt == -1:
			return None
		time.sleep(0.1)
	
	progress.update_state(text = "Calculating Distances...", value = cmax - 1, maximum = cmax)
	
	for proc in procs:
		proc.join()
	for proc in procs:
		proc.terminate()
		proc = None
	
	# distance[i, j] = [diam_dist, ax_dist, d_dist, d_rim_dist], ; where i, j = indices in sample_ids; [name]_dist = components of morphometric distance
	for i in range(profiles_n):
		distance[i,i,:] = 0
	
	for dist in collect_mp:
		mask = (dist != -2)
		distance[mask] = dist[mask]
	
	progress.update_state(text = "Calculating Distances...", value = cmax, maximum = cmax)
	
	return distance

def get_clusters(D, max_clusters = None, limit = 0.68, progress = None):
	# D[i, j] = [diam_dist, ax_dist, d_dist, d_rim_dist], ; where i, j = indices in sample_ids; [name]_dist = components of morphometric distance
	# max_clusters = maximum number of clusters to form
	# limit = maximum morphometric distance between members of one clusters (used only if max_clusters is not specified)
	# 
	# returns clusters, nodes, edges, labels
	# 
	# clusters = {node_idx: [sample_idx, ...], ...}; sample_idx = index in sample_ids
	# nodes = [node_idx, ...]
	# edges = [(source_idx, target_idx), ...]
	# labels = {node_idx: label, ...}
	
	def _calc_dist(cluster1, cluster2, D, d_sq, limit = None):
		
		cluster12 = cluster1 + cluster2
		
		if limit is not None:
			if D[:,cluster12][cluster12].max() > limit:
				return np.inf
		
		# calculate centroid distance
		n1 = len(cluster1)
		n2 = len(cluster2)
		if (n1 == 1) or (n2 == 1):
			d = np.sqrt(d_sq[:,cluster12][cluster12]).mean()
		else:
			d = np.sqrt((np.triu(d_sq[:,cluster12][cluster12]).sum() - (n1 + n2)*(np.triu(d_sq[:,cluster1][cluster1]).sum() / n1 + np.triu(d_sq[:,cluster2][cluster2]).sum() / n2)) / (n1*n2))
		
		return d
	
	def _join_label(label):
		
		return ".".join([str(val) for val in label])
	
	n_samples = D.shape[0]
	
	D[np.diag_indices(n_samples)] = 0
	mask = (D == np.inf)
	D[mask] = D[~mask].max()*2
	mask = (D == 0)
	mins = []
	for idx in range(D.shape[2]):
		d = D[:,:,idx]
		mins.append(d[d > 0].min())
	D = D - mins
	D[mask] = 0
	D = D / D.mean(axis = (0,1))
	
	d_sq = (D**2).sum(axis = 2)
	
	clusters = dict([(idx, [idx]) for idx in range(n_samples)])
	
	to_update = set([])
	d_comb = np.sqrt(d_sq)
	if max_clusters is None:
		d_comb[D.max(axis = 2) > limit] = np.inf
		for idx1, idx2 in np.argwhere(np.triu(d_comb) == np.inf):
			to_update.add((idx1, idx2))
	d_comb[np.diag_indices(n_samples)] = np.inf
	
	cmax = n_samples + 1
	cnt = 1
	
	while (max_clusters is None) or (len(clusters) > max_clusters):
		if progress is None:
			print("\rclustering %d/%d            " % (cnt, cmax), end = "")
		else:
			progress.update_state(text = "Clustering...", value = cnt, maximum = cmax)
			if progress.cancel_pressed():
				return None, None, None, None
		cnt += 1
		
		idx1, idx2 = np.argwhere(d_comb == d_comb.min())[0]
		if d_comb[idx1,idx2] == np.inf:
			break
		d_comb[idx1,idx2] = np.inf
		d_comb[idx2,idx1] = np.inf
		d_comb[idx2] = np.inf
		d_comb[:,idx2] = np.inf
		clusters[idx1] = clusters[idx1] + clusters[idx2]
		del clusters[idx2]
		
		for idx2 in clusters:
			if (idx2 == idx1):
				continue
			if d_comb[idx1,idx2] == np.inf:
				continue
			d_comb[idx1,idx2] = _calc_dist(clusters[idx1], clusters[idx2], D, d_sq, limit if max_clusters is None else None)
			d_comb[idx2,idx1] = d_comb[idx1,idx2]
			if (max_clusters is None) and (d_comb[idx1,idx2] == np.inf):
				to_update.add((idx1, idx2))
	
	if max_clusters is None:
		for idx1, idx2 in to_update:
			if (idx1 in clusters) and (idx2 in clusters):
				d_comb[idx1,idx2] = _calc_dist(clusters[idx1], clusters[idx2], D, d_sq)
				d_comb[idx2,idx1] = d_comb[idx1,idx2]
	
	clusters_lookup = dict([(idx, n_samples + i) for i, idx in enumerate(sorted(clusters.keys()))])
	joined_clusters = dict([(clusters_lookup[idx], clusters_lookup[idx]) for idx in clusters_lookup])
	
	joins = {}
	z_idx = max(clusters_lookup.values()) + 1
	while True:
		if progress is None:
			print("\rclustering %d/%d            " % (cnt, cmax), end = "")
		else:
			progress.update_state(text = "Clustering...", value = cnt, maximum = cmax)
			if progress.cancel_pressed():
				return None, None, None, None
		cnt += 1
		
		idx1, idx2 = np.argwhere(d_comb == d_comb.min())[0]
		if d_comb[idx1,idx2] == np.inf:
			break
		d_comb[idx1,idx2] = np.inf
		d_comb[idx2,idx1] = np.inf
		d_comb[idx2] = np.inf
		d_comb[:,idx2] = np.inf
		
		joins[z_idx] = [joined_clusters[clusters_lookup[idx1]], joined_clusters[clusters_lookup[idx2]]]
		joined_clusters[clusters_lookup[idx1]] = z_idx
		joined_clusters[clusters_lookup[idx2]] = z_idx
		
		z_idx += 1
	z_idx_max = z_idx
	
	clusters = dict([(clusters_lookup[idx], clusters[idx]) for idx in clusters])
	
	edges = set([])
	parents = {}
	for z_idx in joins:
		for idx in joins[z_idx]:
			parents[idx] = z_idx
			edges.add((z_idx, idx))
	for z_idx in clusters:
		for idx in clusters[z_idx]:
			edges.add((z_idx, idx))
	
	nodes = set([])
	for row in edges:
		nodes.update(list(row))
	
	labels = {}
	parent_chains = {}
	for z_idx in joins:
		parent_chains[z_idx] = [z_idx]
		while parent_chains[z_idx][-1] in parents:
			parent_chains[z_idx].append(parents[parent_chains[z_idx][-1]])
		labels[z_idx] = [(z_idx_max - val) for val in parent_chains[z_idx][::-1]]
		for idx_clu in joins[z_idx]:
			labels[idx_clu] = labels[z_idx] + [idx_clu]
	
	d_clu = dict([(z_idx, d_sq[:,clusters[z_idx]][clusters[z_idx]].mean(axis = 0)) for z_idx in clusters])
	
	clusters = dict([(z_idx, sorted(clusters[z_idx], key = lambda idx: d_clu[z_idx][clusters[z_idx].index(idx)])) for z_idx in clusters])
	
	for z_idx in clusters:
		for n, idx in enumerate(clusters[z_idx]):
			labels[idx] = labels[z_idx] + [str(n + 1)]
	
	labels = dict([(idx, _join_label(labels[idx])) for idx in labels])
	
	return clusters, nodes, edges, labels

def update_clusters(D, clusters, progress = None):
	# D[i, j] = [diam_dist, ax_dist, d_dist, d_rim_dist], ; where i, j = indices in sample_ids; [name]_dist = components of morphometric distance
	# clusters = {label: [sample_idx, ...], ...}; sample_idx = index in sample_ids
	# 
	# returns clusters, nodes, edges, labels
	# 
	# clusters = {node_idx: [sample_idx, ...], ...}; sample_idx = index in sample_ids
	# nodes = [node_idx, ...]
	# edges = [(source_idx, target_idx), ...]
	# labels = {node_idx: label, ...}
	
	def _calc_dist(cluster1, cluster2, d_sq):
		
		cluster12 = cluster1 + cluster2
		
		# calculate centroid distance
		n1 = len(cluster1)
		n2 = len(cluster2)
		if (n1 == 1) or (n2 == 1):
			d = np.sqrt(d_sq[:,cluster12][cluster12]).mean()
		else:
			d = np.sqrt((np.triu(d_sq[:,cluster12][cluster12]).sum() - (n1 + n2)*(np.triu(d_sq[:,cluster1][cluster1]).sum() / n1 + np.triu(d_sq[:,cluster2][cluster2]).sum() / n2)) / (n1*n2))
		
		return d
	
	def _join_label(label):
		
		return ".".join([str(val) for val in label])
	
	n_samples = D.shape[0]
	
	D[np.diag_indices(n_samples)] = 0
	mask = (D == np.inf)
	D[mask] = D[~mask].max()*2
	mask = (D == 0)
	mins = []
	for idx in range(D.shape[2]):
		d = D[:,:,idx]
		mins.append(d[d > 0].min())
	D = D - mins
	D[mask] = 0
	D = D / D.mean(axis = (0,1))
	
	d_sq = (D**2).sum(axis = 2)
	
	samples_found = set([])
	for label in clusters:
		samples_found.update(clusters[label])
	clusters = dict([(i, clusters[idx]) for i, idx in enumerate(sorted(clusters.keys()))])
	# make sure each sample is in a cluster
	idx_clu = len(clusters)
	for idx in range(n_samples):
		if idx not in samples_found:
			clusters[idx_clu] = [idx]
			idx_clu += 1
	
	d_comb = np.full((n_samples, n_samples), np.inf, dtype = float)
	for idx1, idx2 in combinations(list(clusters.keys()), 2):
		d_comb[idx1,idx2] = _calc_dist(clusters[idx1], clusters[idx2], d_sq)
		d_comb[idx2,idx1] = d_comb[idx1,idx2]
	
	clusters_lookup = dict([(idx, n_samples + i) for i, idx in enumerate(sorted(clusters.keys()))])
	joined_clusters = dict([(clusters_lookup[idx], clusters_lookup[idx]) for idx in clusters_lookup])
	
	cmax = len(clusters) + 1
	cnt = 1
	
	joins = {}
	z_idx = max(clusters_lookup.values()) + 1
	while True:
		if progress is None:
			print("\rclustering %d/%d            " % (cnt, cmax), end = "")
		else:
			progress.update_state(text = "Clustering...", value = cnt, maximum = cmax)
			if progress.cancel_pressed():
				return None, None, None, None
		cnt += 1
		
		idx1, idx2 = np.argwhere(d_comb == d_comb.min())[0]
		if d_comb[idx1,idx2] == np.inf:
			break
		d_comb[idx1,idx2] = np.inf
		d_comb[idx2,idx1] = np.inf
		d_comb[idx2] = np.inf
		d_comb[:,idx2] = np.inf
		
		joins[z_idx] = [joined_clusters[clusters_lookup[idx1]], joined_clusters[clusters_lookup[idx2]]]
		joined_clusters[clusters_lookup[idx1]] = z_idx
		joined_clusters[clusters_lookup[idx2]] = z_idx
		
		z_idx += 1
	z_idx_max = z_idx
	
	clusters = dict([(clusters_lookup[idx], clusters[idx]) for idx in clusters])
	
	edges = set([])
	parents = {}
	for z_idx in joins:
		for idx in joins[z_idx]:
			parents[idx] = z_idx
			edges.add((z_idx, idx))
	for z_idx in clusters:
		for idx in clusters[z_idx]:
			edges.add((z_idx, idx))
	
	nodes = set([])
	for row in edges:
		nodes.update(list(row))
	
	labels = {}
	parent_chains = {}
	for z_idx in joins:
		parent_chains[z_idx] = [z_idx]
		while parent_chains[z_idx][-1] in parents:
			parent_chains[z_idx].append(parents[parent_chains[z_idx][-1]])
		labels[z_idx] = [(z_idx_max - val) for val in parent_chains[z_idx][::-1]]
		for idx_clu in joins[z_idx]:
			labels[idx_clu] = labels[z_idx] + [idx_clu]
	
	d_clu = dict([(z_idx, d_sq[:,clusters[z_idx]][clusters[z_idx]].mean(axis = 0)) for z_idx in clusters])
	
	clusters = dict([(z_idx, sorted(clusters[z_idx], key = lambda idx: d_clu[z_idx][clusters[z_idx].index(idx)])) for z_idx in clusters])
	
	for z_idx in clusters:
		for n, idx in enumerate(clusters[z_idx]):
			labels[idx] = labels[z_idx] + [str(n + 1)]
	
	labels = dict([(idx, _join_label(labels[idx])) for idx in labels])
	
	return clusters, nodes, edges, labels

