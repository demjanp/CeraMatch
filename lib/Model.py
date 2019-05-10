# -*- coding: utf-8 -*- 

from deposit import (Store, Commander, Broadcasts)

from fnc_matching import *

from PySide2 import (QtCore, QtGui)
from natsort import (natsorted)
import numpy as np
import json

FSAMPLE_IDS = "data/matching_find_ids.json"
FMATCHING = "data/matching.npy"
DB_URL = "c:\\Documents\\AE_KAP_morpho\\db_typology\\kap_typology.json"

class Model(Store):
	
#	ARROW_LEFT = "\u2b98"
#	ARROW_LEFT = "\u2190"
	ARROW_LEFT = "\u2b9c"
#	ARROW_RIGHT = "\u2b9a"
#	ARROW_RIGHT = "\u2192"
	ARROW_RIGHT = "\u2b9e"
	BULLET = "\u25cf"
	
	def __init__(self, view):
		
		self.view = view
		self.dc = None
		self.sample_data = []  # [[sample_id, DResource, label, value], ...]
		self.sample_ids = []
		self.distance = None
		self.weights = {
			"Radius": 0,
			"Tangent": 0,
			"Curvature": 0,
			"Hamming": 0,
			"Diameter": 0,
			"Axis": 0,
		}
		for name in self.weights:
			self.weights[name] = 1/len(self.weights)
		
		Store.__init__(self, parent = self.view)
		
		self.load(DB_URL)

		with open(FSAMPLE_IDS, "r") as f:
			self.sample_ids = json.load(f)
		
		for id in self.classes["Sample"].objects:
			
			obj = self.objects[id]
			
			sample_id = obj.descriptors["Id"].label.value
			if sample_id not in self.sample_ids:
				continue
			self.sample_data.append([sample_id, obj.descriptors["Reconstruction"].label, sample_id, sample_id])
		
		self.distance = np.load(FMATCHING) # distance[i, j] = [R_dist, th_dist, kap_dist, h_dist, diam_dist, ax_dist]
	
	def set_weight(self, name, value):
		
		if name not in self.weights:
			return
		self.weights[name] = value
		names2 = [name2 for name2 in self.weights if name2 != name]
		values = np.array([self.weights[name2] for name2 in names2])
		sum_tgt = 1 - value
		if values.sum() > 0:
			values *= sum_tgt / values.sum()
		for idx, name2 in enumerate(names2):
			self.weights[name2] = float(values[idx])
			self.view.sliders_frame.set_slider(name2, int(round(self.weights[name2] * 100)))
	
	def load_clustering(self):
		
		def hamming_name(name1, name2):
			
			name1 = name1.split(".")
			name2 = name2.split(".")
			min_lenght = min(len(name1), len(name2))
			for i in range(min_lenght):
				if name1[i] != name2[i]:
					break
			return 1 - (2*i) / (len(name1) + len(name2))
		
		D = combine_dists(self.distance, self.weights["Radius"], self.weights["Tangent"], self.weights["Curvature"], self.weights["Hamming"], self.weights["Diameter"], self.weights["Axis"])
		clusters = get_max_clusters(D)
		
		names = natsorted(clusters.keys())
		
		collect = [] # [[idx, cluster_name], ...]
		for idx in clusters[names[0]]:
			collect.append([idx, names[0]])
		idx_last = collect[-1][0]
		for name in names[1:]:
			for i in np.argsort(D[idx_last][clusters[name]]):
				collect.append([clusters[name][i], name])
		
		data = {} # {sample_id: [caption, cluster name, order], ...}
		idx, name = collect[0]
		data[self.sample_ids[idx]] = [None, name, i]
		for i in range(1, len(collect)):
			_, name_left = collect[i - 1]
			idx, name = collect[i]
			h = hamming_name(name, name_left)
			color = 255 - int(round(h*255))
			label = QtGui.QColor(color, color, color, 255)
			data[self.sample_ids[idx]] = [label, name, i]
		
		self.update_captions(data)
	
	def load_distance(self, sample_id):
		
		D = combine_dists(self.distance, self.weights["Radius"], self.weights["Tangent"], self.weights["Curvature"], self.weights["Hamming"], self.weights["Diameter"], self.weights["Axis"])
		
		data = {} # {sample_id: [rounded distance, distance, order], ...}
		idx0 = self.sample_ids.index(sample_id)
		idxs = np.argsort(D[idx0])
		i = 0
		for idx1 in idxs:
			d = D[idx0,idx1]
			data[self.sample_ids[idx1]] = [str(round(d, 2)), d, i]
			i += 1
		self.update_captions(data)
		self.view.image_lst.scrollToTop()
	
	def load_ids(self):
		
		for i in range(len(self.sample_data)):
			self.sample_data[i][2] = self.sample_data[i][0]
		self.view.image_lst.reload()
	
	def update_captions(self, data):
		# data = {sample_id: [label, value, order], ...}
		
		ordered = []
		for idx in range(len(self.sample_data)):
			sample_id, resource, _, _ = self.sample_data[idx]
			label, value, order = data[sample_id]
			ordered.append([order, sample_id, resource, label, value])
		self.sample_data = [row[1:] for row in sorted(ordered, key = lambda row: row[0])]
		self.view.image_lst.reload()
	
	def set_datasource(self, data_source):
		
		Store.set_datasource(self, data_source)
	
	def get_temp_dir(self):
		
		tempdir = os.path.normpath(os.path.abspath(os.path.join(tempfile.gettempdir(), "lap")))
		if not os.path.exists(tempdir):
			os.makedirs(tempdir)
		return tempdir
	
	def launch_commander(self):
		
		self.dc = Commander(model = self)

