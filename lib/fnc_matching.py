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

def rotate_coords(coords, angle):
	
	return np.dot(coords,np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]]))

def smoothen_coords(coords, step = 0.5, resolution = 0.5):
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

def clean_mask(mask):
	
	labels = measure.label(mask, background = False)
	values = np.unique(labels)
	values = values[values > 0]
	value = sorted(values.tolist(), key = lambda value: (labels == value).sum())[-1]
	return (labels == value)

def profile_axis(profile, rasterize_factor):
	
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
	
	collect = []
	for size in [3,5]:
		collect.append(np.zeros((size,size), dtype = bool))
		collect[-1][0] = True
		collect[-1][-1] = True
		collect[-1][:,0] = True
		collect[-1][:,-1] = True
		collect[-1] = np.argwhere(collect[-1])
	around1 = collect[0] - 1
	around2 = collect[1] - 2
	
	mask = skeletonize(profile_mask)
	nodes = np.zeros(mask.shape, dtype = bool)
	skelet = mask.copy()
	for i, j in np.argwhere(mask):
		ijs = around1 + [i,j]
		ijs = ijs[(ijs >= 0).all(axis = 1) & (ijs < mask.shape).all(axis = 1)]
		if mask[ijs[:,0], ijs[:,1]].sum() == 3:
			ijs = around2 + [i,j]
			if mask[ijs[:,0], ijs[:,1]].sum() == 3:
				skelet[i,j] = False
				nodes[i,j] = True
	nodes = dilation(nodes, square(3))
	skelet[nodes] = False
	collect = set([])
	for i, j in np.argwhere(nodes):
		ijs = around1 + [i,j]
		ijs = ijs[(ijs >= 0).all(axis = 1) & (ijs < skelet.shape).all(axis = 1)]
		ijs = ijs[skelet[ijs[:,0], ijs[:,1]]]
		for i, j in ijs:
			collect.add((i, j))
	endpoints = np.array(list(collect), dtype = int)
	if not endpoints.size:
		for i, j in np.argwhere(mask):
			ijs = around1 + [i,j]
			ijs = ijs[(ijs >= 0).all(axis = 1) & (ijs < mask.shape).all(axis = 1)]
			if mask[ijs[:,0], ijs[:,1]].sum() == 1:
				endpoints = np.array([[i,j]], dtype = int)
				break
	
	paths = []
	done = set([])
	for i0, j0 in endpoints:
		if (i0, j0) in done:
			continue
		path = [[i0,j0]]
		i, j = path[-1]
		while True:
			done.add((i, j))
			ijs = around1 + [i,j]
			ijs = ijs[(ijs >= 0).all(axis = 1) & (ijs < skelet.shape).all(axis = 1)]
			slice = ijs[skelet[ijs[:,0], ijs[:,1]]]
			found = False
			for i1, j1 in slice:
				if (i1, j1) not in done:
					path.append([i1, j1])
					done.add((i1, j1))
					found = True
					break
			if not found:
				break
			i, j = path[-1]
		paths.append(path)
	
	paths = sorted(paths, key = lambda path: len(path))
	
	axis = np.array(paths.pop(), dtype = float)	
	
	if ((axis[-1] / rasterize_factor - [-x0, -y0])**2).sum() < ((axis[0] / rasterize_factor - [-x0, -y0])**2).sum():
		axis = axis[::-1]
	
	d_max = rasterize_factor * 2
	while paths:
		path = paths.pop()
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
	
	d = cdist(axis, profile).min(axis = 1)
	thickness = d * 2
	thickness = thickness[thickness > thickness.max() / 2].mean()
	d = cdist(axis, [profile[0], profile[-1]]).min(axis = 1)
	axis = axis[d > thickness]
	
	d = cdist([axis[0]], axis)[0]
	axis = axis[d > thickness / 2]
	
	axis = axis / rasterize_factor + [x0, y0]
	
	axis = smoothen_coords(axis)
	
	return axis

def diameter_dist(axis1, radius1, axis2, radius2):
	
	axis1 = axis1 + [radius1, 0]
	axis2 = axis2 + [radius2, 0]
	ymax = min(axis1[:,1].max(), axis2[:,1].max())
	axis1 = axis1[axis1[:,1] < ymax]
	axis2 = axis2[axis2[:,1] < ymax]
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

def find_keypoints(profile, rasterize_factor):
	
	[x0, y0], [x1, y1] = profile.min(axis = 0), profile.max(axis = 0)
	x0, y0, x1, y1 = int(x0) - 1, int(y0) - 1, int(x1) + 1, int(y1) + 1
	w, h = x1 - x0, y1 - y0
	
	img = Image.new("1", (w * rasterize_factor, h * rasterize_factor))
	draw = ImageDraw.Draw(img)
	profile_r = ((profile - [x0, y0]) * rasterize_factor).astype(int)
	draw.polygon([(x, y) for x, y in profile_r], fill = 0, outline = 1)
	outline = np.argwhere(np.array(img, dtype = bool).T)
	draw.polygon([(x, y) for x, y in profile_r], fill = 1, outline = 0)
	profile_mask = clean_mask(np.array(img, dtype = bool).T)
	img.close()
	
	axis = np.argwhere(skeletonize(profile_mask)) / rasterize_factor + [x0, y0]
	
	d = cdist(axis, profile)
	thickness = d.min(axis = 1) * 2
	thickness = thickness[thickness > thickness.max() / 2].mean()
	d = cdist(axis, [profile[0], profile[-1]]).min(axis = 1)
	mask = (d > thickness)
	if mask.any():
		axis = axis[mask]
	else:
		axis = np.array([profile.mean(axis = 0)])
	
	d_axis_profile = cdist(axis, profile)
	thickness = d_axis_profile.min(axis = 1) * 2
	thickness = thickness[thickness > thickness.max() / 2].mean()
	
	length = np.sqrt((np.diff(profile, axis = 0) ** 2).sum(axis = 1)).sum() / 2
	
	step = min(thickness / 2, length * 0.05)
	
	d_profile = squareform(pdist(profile))
	d_axis = squareform(pdist(axis))
	d_profile[d_profile == 0] = np.inf
	keypoints = np.zeros(axis.shape[0])
	for i in range(profile.shape[0]):
		idxs = np.where(d_profile[i] < step)[0]
		if not idxs.size:
			continue
		i0 = idxs.min()
		i1 = idxs.max()
		if i0 >= i1:
			continue
		angle = np.arctan2(*(profile[i1] - profile[i0])[::-1])
		if angle > 0:
			angle = angle + np.pi
		angle += np.pi / 2
		segment = rotate_coords(profile[i0:i1], -angle).astype(float)
		segment -= segment[0]
		k = np.argmax(np.abs(segment[:,0]))
		val = abs(segment[k,0])
		
		idx_profile = k + i0
		idx_axis = np.argmin(d_axis_profile[:,idx_profile])
		
		idxs_axis = np.where(d_axis[idx_axis] < 4*step)
		if val > keypoints[idxs_axis].max():
			keypoints[idxs_axis] = 0
			keypoints[idx_axis] = val
	
	keypoints = np.where(keypoints > 0)[0]
	
	return axis[keypoints], thickness

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
	distance = np.zeros((profiles_n, profiles_n, 3), dtype = float) - 2
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
		
		distance[i, j, 0] = diam_dist
		distance[i, j, 1] = ax_dist
		distance[i, j, 2] = h_dist
		
		distance[j, i, 0] = diam_dist
		distance[j, i, 1] = ax_dist
		distance[j, i, 2] = h_dist
		
	collect_mp.append(distance)
	
def calc_distances(profiles, distance = None):
	# profiles[sample_id] = [profile, radius]
	# returns distance[i, j] = [diam_dist, ax_dist, h_dist], ; where i, j = indices in sample_ids; [name]_dist = components of morphometric distance
	
	rasterize_factor = 10
	
	profiles_n = len(profiles)
	sample_ids = list(profiles.keys())
	
	manager = mp.Manager()
	if distance is None:
		distance = np.ones((profiles_n, profiles_n, 3), dtype = float)
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
		axis = profile_axis(profile, rasterize_factor)
		axis = axis - axis[0]
		keypoints, thickness = find_keypoints(profile, rasterize_factor)
		data[sample_id] = [profile, axis, radius, keypoints, thickness]
	collect_mp = manager.list()
	cmax = len(ijs_mp)
	procs = []
	for pi in range(mp.cpu_count()):
		procs.append(mp.Process(target = dist_worker, args = (ijs_mp, collect_mp, data, sample_ids, cmax)))
		procs[-1].start()
	for proc in procs:
		proc.join()
	for proc in procs:
		proc.terminate()
		proc = None
	
	# distance[i, j] = [diam_dist, ax_dist, h_dist], ; where i, j = indices in sample_ids; [name]_dist = components of morphometric distance
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

def get_clusters(D, limit = 0.68, hierarchical_labels = True):
	# D[i, j] = [diam_dist, ax_dist, h_dist], ; where i, j = indices in sample_ids; [name]_dist = components of morphometric distance
	# returns clusters = {label: [i, ...], ...}; i = index in sample_ids
	
	def _calc_dist(cluster, D, limit = None):
		
		d = D[:,cluster][cluster]
		if (limit is not None) and (d.max() > limit):
			return np.inf
		return d.mean()
	
	mask = (D == 0)
	mins = []
	for idx in range(D.shape[2]):
		d = D[:,:,idx]
		mins.append(d[d > 0].min())
	D = D - mins
	D[mask] = 0
	D = D / D.mean(axis = (0,1))
	
	limit = D.mean() * limit
	
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
	
	if not hierarchical_labels:
		return dict([(str(ci), clusters[label]) for ci, label in enumerate(list(clusters.keys()))])
	
	# hierarchically label clusters
	
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

