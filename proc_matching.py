from fnc_matching import *
import json
import os
import numpy as np

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
		radius = np.array(profiles[sample_id]["radius"])
		params = profiles[sample_id]["break_params"]
		collect[sample_id] = [profile, radius, params]
	profiles = collect
	
	profiles_n = len(profiles)
	sample_ids = list(profiles.keys())
	
	with open(ffindids, "w") as f:
		json.dump(sample_ids, f)
	
	distance = calc_distances(profiles)
	
	# normalize by ensemble average (see Karasik and Smilansky 2011)
	avg_R, avg_th, avg_kap = 0, 0, 0
	for sample_id in profiles:
		prof = profiles[sample_id][0] + [profiles[sample_id][1], 0]
		L = np.abs(arc_length(prof)).sum()
		avg_R += np.sqrt((prof[1:,0]**2).sum() / L)
		th = fftsmooth(tangent(prof))
		avg_th += np.sqrt((th**2).sum() / L)
		avg_kap += np.sqrt((np.hstack(([0], np.diff(th)))**2).sum() / L)
	M = len(profiles)
	avg_R, avg_th, avg_kap = avg_R / M, avg_th / M, avg_kap / M
	distance[:,:,0] /= avg_R
	distance[:,:,1] /= avg_th
	distance[:,:,2] /= avg_kap
	
	np.save(fmatching, distance)
	# distance[i, j] = [R_dist, th_dist, kap_dist, h_dist, diam_dist, ax_dist]

