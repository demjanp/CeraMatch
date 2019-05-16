from deposit import Store
from lib.fnc_matching import *

from skimage.measure import find_contours
from scipy.spatial.distance import cdist
from PIL import Image, ImageDraw
import numpy as np
import json

BREAK_SHIFT = 5 # offset of breaklines from profile outline (in mm)
RASTERIZE_FACTOR = 50

fprofiles = "data/profiles.json"

def breakline_params(points):
	
	(x0, y0), (x1, y1) = points
	if x0 != x1:
		a = (y1 - y0) / (x1 - x0)
	else:
		a = 0
	b = y0 - a * x0
	if x0 != x1:
		x1 = x0 - 1 / np.sqrt(a**2 + 1)
		y1 = a * x1 + b
	else:
		x1 = x0
		y1 = y0 + 1
	xd, yd = x1 - x0, y1 - y0
	
	x1, y1 = points[1]
	x2, y2 = x0 + xd, y0 + yd
	x3, y3 = x0 - xd, y0 - yd
	if abs(x1 - x2) < abs(x1 - x3):
		xd = -xd
	if abs(y1 - y2) < abs(y1 - y3):
		yd = -yd
	
	return x0, y0, xd, yd

def extend_breakline(points):
	
	x0, y0, xd, yd = breakline_params(points)
	x1, y1 = x0 + BREAK_SHIFT * xd, y0 + BREAK_SHIFT * yd
	
	return x1, y1

store = Store()
store.load("c:\\Documents\\AE_KAP_morpho\\db_typology\\kap_typology.json")

collect = {} # {sample_id: {rim, bottom, oriented, radius, profile, arc, breaks, break_params}, ...}
cmax = len(store.classes["Sample"].objects)
cnt = 1
for id in store.classes["Sample"].objects:
	print("\rprocessing %d/%d          " % (cnt, cmax), end = "")
	cnt += 1
	
	obj = store.objects[id]
	
	sample_id = obj.descriptors["Id"].label.value
	profile = np.array(obj.descriptors["Profile"].label.coords[0])
	bottom = bool(int(obj.descriptors["Bottom"].label.value))
	radius = obj.descriptors["Radius"].label.value
	norm_length = float(obj.descriptors["Norm_Length"].label.value)
	thickness = float(obj.descriptors["Thickness"].label.value)
	
	if (not bottom) and (norm_length < 6):
		continue
	
	if radius:
		radius = float(radius)
	arc = []
	breaks = []
	for id2 in obj.relations["Drawn"]:
		obj2 = store.objects[id2]
		if "Arc" in obj2.classes:
			arc = obj2.descriptors["Geometry"].label.coords[0]
		if "Break" in obj2.classes:
			break1 = np.array(obj2.descriptors["Break_1"].label.coords[0])
			break2 = np.array(obj2.descriptors["Break_2"].label.coords[0])
			breaks.append([break1, break2])
	
	if ":" in sample_id:
		continue
	if ".." in sample_id:
		continue
	if not radius:
		continue
	if not ((len(breaks) == 1) or bottom):
		continue
	
	# shift profile so that [0,0] is at the rim
	profile = profile - get_rim(profile)
	
	# find and remove break from profile
	# find parameters to extend profile axis
	params = []
	if breaks:
		breaks = breaks[0]
		break1, break2 = breaks
		xb1, yb1 = extend_breakline(break1)
		xb2, yb2 = extend_breakline(break2)
		d = cdist([[xb1, yb1]], profile)[0]
		b1 = np.argmin(d)
		d = cdist([[xb2, yb2]], profile)[0]
		b2 = np.argmin(d)
		if b1 > b2:
			b1, b2 = b2, b1
		x0, y0 = (profile[b1] + profile[b2]) / 2
		profile1 = profile[b1:b2+1]
		profile2 = np.vstack((profile[b2:], profile[:b1+1]))
		if profile_length(profile1) > profile_length(profile2):
			profile = profile1
		else:
			profile = profile2
		_, _, xd, yd = np.vstack((breakline_params(break1), breakline_params(break2))).T.mean(axis = 1)
		xd, yd = -xd, -yd
		params = [x0, y0, xd, yd]
		breaks = [br.tolist() for br in breaks]
		
	else:
		profile = set_idx0_to_point(profile, np.array([0,0]))
		idx = np.argmax((profile**2).sum(axis = 1))
		profile = np.vstack((profile[idx:], profile[:idx]))
	
	[x0, y0], [x1, y1] = profile.min(axis = 0), profile.max(axis = 0)
	x0, y0, x1, y1 = int(x0) - 1, int(y0) - 1, int(x1) + 1, int(y1) + 1
	w, h = (x1 - x0) + 2, (y1 - y0) + 2
	
	img = Image.new("1", (w * RASTERIZE_FACTOR, h * RASTERIZE_FACTOR))
	draw = ImageDraw.Draw(img)
	
	draw.polygon([(x + 1, y + 1) for x, y in ((profile - [x0, y0]) * RASTERIZE_FACTOR).astype(int)], fill = 1)
	contours = find_contours(np.array(img, dtype = bool).T, 0)
	contours = sorted(contours, key = lambda contour: len(contour))
	outline = contours[-1] / RASTERIZE_FACTOR
	
	img.close()
	
	profile -= profile.min(axis = 0)
	outline -= outline.min(axis = 0)
	
	collect_outline = []
	step = 0.2
	for point in outline:
		if (not collect_outline) or (cdist([point], [collect_outline[-1]])[0][0] >= step):
			collect_outline.append(point)
	outline = np.array(collect_outline)
	
	outline = set_idx0_to_point(outline, (profile[0] + profile[-1]) / 2)
	
	idx0, idx1 = 0, 0
	di = 0
	while idx0 == idx1:
		idx0 = np.argmin(cdist([profile[di]], outline)[0])
		idx1 = np.argmin(cdist([profile[-(1+di)]], outline)[0])
		di += 1
	if idx0 > idx1:
		idx0, idx1 = idx1, idx0
	
	dx = np.gradient(outline[:,0])
	dy = np.gradient(outline[:,1])
	ds = np.sqrt(dx**2 + dy**2)
	if ds[idx0:idx1].sum() < ds[:idx0].sum() + ds[idx1:].sum():
		outline = np.vstack((outline[:idx0], outline[idx1:]))
	else:
		outline = outline[idx0:idx1]
	
	profile = fftsmooth(outline, params = params)
	
	profile = profile - get_rim(profile)
	idx_rim = np.argmin(cdist([[0,0]], profile)[0])
	if profile[idx_rim + 1,0] < profile[idx_rim - 1,0]:
		profile = profile[::-1]
		profile -= get_rim(profile)
	
	collect[sample_id] = dict(
		bottom = bottom,
		radius = radius,
		thickness = thickness,
		profile = profile.tolist(),
		arc = arc,
		break_params = params,
	)

print()
print()
print("extracted: %d profiles" % (len(collect)))
print()

with open(fprofiles, "w") as f:
	json.dump(collect, f)

