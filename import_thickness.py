from deposit import Store

import json
import numpy as np
from scipy.spatial.distance import cdist
from skimage.morphology import thin
from PIL import Image, ImageDraw

fprofiles = "data/profiles.json"

def profile_to_raster_and_outline(profile, rasterize_factor):

	[x0, y0], [x1, y1] = profile.min(axis = 0), profile.max(axis = 0)
	x0, y0, x1, y1 = int(x0) - 1, int(y0) - 1, int(x1) + 1, int(y1) + 1
	w, h = x1 - x0, y1 - y0
	
	profile = ((profile - [x0, y0]) * rasterize_factor).astype(int)
	
	img = Image.new("1", (w * rasterize_factor, h * rasterize_factor))
	draw = ImageDraw.Draw(img)
	draw.polygon([(x, y) for x, y in profile], fill = 1)
	raster = np.array(img, dtype = bool).T
	img.close()
	
	img = Image.new("1", (w * rasterize_factor, h * rasterize_factor))
	draw = ImageDraw.Draw(img)
	draw.polygon([(x, y) for x, y in profile], fill = 0, outline = 1)
	outline = np.array(img, dtype = bool).T
	img.close()
	
	return raster, outline

store = Store()
store.load("c:\\Documents\\AE_KAP_morpho\\db_typology\\kap_typology.json")

id_lookup = {} # {Sample.Id: obj_id, ...}
for id in store.classes["Sample"].objects:
	id_lookup[store.objects[id].descriptors["Id"].label.value] = id

with open(fprofiles, "r") as f:
	profiles = json.load(f)

rasterize_factor = 5

cmax = len(profiles)
cnt = 1
for sample_id in profiles:
	
	print("\r%d/%d          " % (cnt, cmax), end = "")
	cnt += 1
	
	profile = np.array(profiles[sample_id]["profile"])
	
	raster, outline = profile_to_raster_and_outline(profile, rasterize_factor)
	thinned = thin(raster)
	
	thinned = np.argwhere(thinned)
	outline = np.argwhere(outline)
	thickness = ((cdist(thinned, outline).min(axis = 1) * 2) / rasterize_factor)
	thickness = thickness[thickness > thickness.max() / 2].mean()
	
	store.objects[id_lookup[sample_id]].add_descriptor("Thickness", thickness)

store.save()
