from lib.fnc_matching import *
import json
import os
import numpy as np
from scipy import spatial, optimize

fprofiles = "data/profiles.json"
ffindids = "data/matching_find_ids.json"
fmatching = "data/matching.npy"

if __name__ == '__main__':
	
	with open(fprofiles, "r") as f:
		profiles = json.load(f)
	
	# profiles = {sample_id: {
	#	rim: True/False, 
	#	bottom: True/False, 
	#	oriented: True/False, 
	#	radius: float, 
	#	thickness: float, 
	#	profile: [[x, y], ...], 
	#	arc: [[x, y], ...],
	#	breaks: [[[x, y], ...], [[x, y], ...]],
	#	break_params: [x0, y0, xd, yd],
	# 		x1 = x0 + xd * length
	# 		y1 = y0 + yd * length; where length = length of the resulting line extending the profile 
	#}, ...}
	
	collect = {}
	for sample_id in profiles:
		profile = np.array(profiles[sample_id]["profile"])
		bottom = np.array(profiles[sample_id]["bottom"])
		radius = np.array(profiles[sample_id]["radius"])
		thickness = np.array(profiles[sample_id]["thickness"])
		params = profiles[sample_id]["break_params"]
		collect[sample_id] = [profile, bottom, radius, thickness, params]
	profiles = collect
	
	profiles_n = len(profiles)
	sample_ids = list(profiles.keys())
	
	print("found %d profiles" % (profiles_n))
	
	with open(ffindids, "w") as f:
		json.dump(sample_ids, f)
	
	distance = calc_distances(profiles)
	
	np.save(fmatching, distance)
	# distance[i, j] = [diam_dist, thick_dist, ax_dist, h_dist, ht_dist]
