from deposit import Store

import json
import numpy as np
from scipy import spatial, optimize
import psycopg2

fprofiles = "data/profiles.json"

def match_circle(coords):

	def f_dist(c):
		Ri = spatial.distance.cdist([c], coords)[0]
		return Ri - Ri.mean()
	
	center = optimize.leastsq(f_dist, coords.mean(axis = 0))[0]
	Ri = spatial.distance.cdist([center], coords)[0]
	r = Ri.mean()
	residu = (Ri - r).std()
	
	return center, r, residu

with open(fprofiles, "r") as f:
	profiles = json.load(f)

# profiles = {find_id: {
#	rim: True/False, 
#	bottom: True/False, 
#	oriented: True/False, 
#	radius: float, 
#	profile: [[x, y], ...], 
#	arc: [[x, y], ...],
#}, ...}

store = Store()
store.load("c:\\Documents\\AE_KAP_morpho\\db_typology\\kap_typology.json")

id_lookup = {} # {Sample.Id: obj_id, ...}
for id in store.classes["Sample"].objects:
	id_lookup[store.objects[id].descriptors["Id"].label.value] = id

cmax = len(id_lookup)
cnt = 1
for sample_id in profiles:
	print("\r%d/%d          " % (cnt, cmax), end = "")
	cnt += 1
	arc = np.array(profiles[sample_id]["arc"])
	if arc.size:
		l = np.sqrt((np.diff(arc, axis = 0)**2).sum(axis = 1)).sum()
		_, r, residu = match_circle(arc)
		error = residu / r
		c = 2 * np.pi * r
		store.objects[id_lookup[sample_id]].add_descriptor("EVE", round(l / c, 4))

store.save()
