from deposit import (Store, Commander, Broadcasts)

from lib.fnc_matching import *
from lib.Sample import Sample

from PySide2 import (QtCore, QtGui)
from natsort import (natsorted)
import numpy as np
import json

FSAMPLE_IDS = "data/matching_find_ids.json"
FMATCHING = "data/matching.npy"
DB_URL = "c:\\Documents\\AE_KAP_morpho\\db_typology\\kap_typology.json"

class Model(Store):
	
	def __init__(self, view):
		
		self.view = view
		self.dc = None
		self.samples = [] # [Sample, ...]
		self.sample_ids = []
		self.distance = None
		self.max_clusters = None
		self.weights = {
			"Radius": 0,
			"Tangent": 0,
			"Curvature": 0,
			"Hamming": 0,
			"Diameter": 0,
			"Axis": 0,
		}
		self.cluster_weights = {}  # {cluster_name: weights, ...}
		self.subcluster_numbers = {}  # {cluster_name: subcluster number, ...}
		for name in self.weights:
			self.weights[name] = 1/len(self.weights)
		
		
		Store.__init__(self, parent = self.view)
		
		self.load(DB_URL)

		with open(FSAMPLE_IDS, "r") as f:
			self.sample_ids = json.load(f)
		
		row = 0
		for id in self.classes["Sample"].objects:
			obj = self.objects[id]
			sample_id = obj.descriptors["Id"].label.value
			if sample_id not in self.sample_ids:
				continue
			self.samples.append(Sample(sample_id, obj.descriptors["Reconstruction"].label, sample_id, sample_id, row))
			row += 1
		
		self.distance = np.load(FMATCHING) # distance[i, j] = [R_dist, th_dist, kap_dist, h_dist, diam_dist, ax_dist]
		
		self.max_clusters = len(self.sample_ids) // 2
	
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
			self.view.weights_frame.set_slider(name2, int(round(self.weights[name2] * 100)))
	
	def get_clustering(self, sample_ids, clusters_n, color = 3, gradient = False):
		# color: 1 = green, 2 = blue, 3 = greyscale
		# gradient: if True, assign color gradient based on difference from cluster of previous sample, else assign alternating blue / green to clusters
		# returns 
		#	data = {sample_id: [caption, cluster_name, order], ...}
		#	names = [cluster_name, ...]
		
		def hamming_name(name1, name2):
			
			name1 = name1.split("-")[-1].split(".")
			name2 = name2.split("-")[-1].split(".")
			min_lenght = min(len(name1), len(name2))
			for i in range(min_lenght):
				if name1[i] != name2[i]:
					break
			return 1 - (2*i) / (len(name1) + len(name2))
		
		def get_ordered_clusters(sample_ids, clusters_n):
			# returns [[sample_id, cluster_name, order], ...]
			
			sample_idxs = sorted([self.sample_ids.index(sample_id) for sample_id in sample_ids])
			D = combine_dists(self.distance[:,sample_idxs][sample_idxs], self.weights["Radius"], self.weights["Tangent"], self.weights["Curvature"], self.weights["Hamming"], self.weights["Diameter"], self.weights["Axis"])
			clusters = get_n_clusters(D, clusters_n)
			names = natsorted(clusters.keys())
			
			collect = []  # [sample_id, cluster_name, order]
			order = 0
			for idx in clusters[names[0]]:
				collect.append([self.sample_ids[sample_idxs[idx]], names[0], order])
				order += 1
			for name in names[1:]:
				for idx in clusters[name]:
					collect.append([self.sample_ids[sample_idxs[idx]], name, order])
					order += 1
			return collect, names
		
		sample_ids = sample_ids.copy()
		
		clusters, names = get_ordered_clusters(sample_ids, clusters_n)  # [[sample_id, cluster_name, order], ...]
		
		data = {} # {sample_id: [caption, cluster name, order], ...}
		
		if gradient:
			for i in range(len(clusters)):
				sample_id, name, order = clusters[i]
				if i == 0:
					name_left = name
				else:
					_, name_left, _ = clusters[i - 1]
				h = hamming_name(name, name_left)
				grad = 255 - int(round(h*255))
				if color == 3:
					rgb = [grad, grad, grad, 255]
				else:
					rgb = [255,255,255,255]
					for cidx in range(3):
						if cidx != color:
							rgb[cidx] = grad
				data[sample_id] = [QtGui.QColor(*rgb), name, order]
		
		else:
			if color == 3:
				colors = [QtGui.QColor(200, 200, 200, 255), QtGui.QColor(128, 128, 128, 255)]
			else:
				colors = [QtGui.QColor(200, 255, 200, 255), QtGui.QColor(200, 200, 255, 255)]
			ci = True
			name_prev = None
			for i in range(len(clusters)):
				sample_id, name, order = clusters[i]
				if name != name_prev:
					ci = not ci
					name_prev = name
				label = colors[int(ci)]
				data[sample_id] = [label, name, order]
		
		return data, names
	
	def load_clustering(self, clusters_n):
		
		data = {} # {sample_id: [caption, cluster_name, order], ...}
		colors = {}
		if clusters_n < self.max_clusters:
			data, names = self.get_clustering(self.sample_ids, clusters_n, color = 1, gradient = False)
			for i, name in enumerate(names):
				sample_ids = [sample_id for sample_id in self.sample_ids if data[sample_id][1] == name]
				subclusters_n = len(sample_ids) // 2
				order0 = min([data[sample_id][2] for sample_id in sample_ids])
				if subclusters_n > 1:
					color = 1 if (i % 2 == 0) else 2
					subdata, _ = self.get_clustering(sample_ids, subclusters_n, color = color, gradient = True)
					for sample_id in subdata:
						colors[sample_id] = color
						caption, name, order = subdata[sample_id]
						data[sample_id][0] = caption
						data[sample_id][1] = "%s-%s" % (data[sample_id][1], name)
						data[sample_id][2] = order0 + order
				else:
					color = [QtGui.QColor(200, 255, 200, 255), QtGui.QColor(200, 200, 255, 255)][i % 2]
					for sample_id in sample_ids:
						colors[sample_id] = 1 if (i % 2 == 0) else 2
						data[sample_id][0] = color
						data[sample_id][1] = "%s-1" % (data[sample_id][1])
		else:
			data, names = self.get_clustering(self.sample_ids, clusters_n, color = 3, gradient = True)
		
		self.cluster_weights = {}
		self.subcluster_numbers = {}
		if clusters_n < self.max_clusters:
			for sample in self.samples:
				sample.cluster = data[sample.id][1].split("-")[0]
				sample.cluster_color = colors[sample.id]
			self.cluster_weights["main"] = dict([(weight_name, self.weights[weight_name]) for weight_name in self.weights])
		
		self.update_captions(data)
	
	def load_subclustering(self, clusters_n, color = 3, gradient = False, store_subcluster = True):
		
		selected = self.view.get_selected()
		if not selected:
			return
		cluster = selected[0].cluster
		if cluster is None:
			return
		samples = dict([(sample.id, sample) for sample in self.samples if sample.cluster == cluster])
		if clusters_n is None:
			clusters_n = len(samples) // 2
		self.subcluster_numbers[cluster] = clusters_n
		data, _ = self.get_clustering(list(samples.keys()), clusters_n, color = color, gradient = gradient)
		order0 = min([samples[sample_id].row for sample_id in samples])
		names = set()
		for sample_id in data:
			label, value, order = data[sample_id]
			samples[sample_id].label = label
			samples[sample_id].value = "%s-%s" % (cluster, value)
			samples[sample_id].row = order0 + order
			if store_subcluster:
				samples[sample_id].subcluster = value
				names.add(samples[sample_id].value)
			else:
				samples[sample_id].subcluster = None
		self.samples = sorted(self.samples, key = lambda sample: sample.row)
		if store_subcluster:
			for name in names:
				self.cluster_weights[name] = dict([(weight_name, self.weights[weight_name]) for weight_name in self.weights])
		self.view.image_lst.reload()
	
	def clear_subclustering(self):
		
		selected = self.view.get_selected()
		if not selected:
			return
		cluster = selected[0].cluster
		if cluster is None:
			return
		color = selected[0].cluster_color
		if color is None:
			color = 3
		self.load_subclustering(None, color = color, gradient = True, store_subcluster = False)
	
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
		for sample in self.samples:
			sample.cluster = None
		
		self.update_captions(data)
		self.view.image_lst.scrollToTop()
	
	def load_ids(self):
		
		data = {} # {sample_id: [label, value, order], ...}
		sorted_ids = natsorted(self.sample_ids)
		for i in range(len(sorted_ids)):
			sample_id = sorted_ids[i]
			data[sample_id] = [sample_id, sample_id, i]
		self.update_captions(data)
		for sample in self.samples:
			sample.cluster = None
		self.view.image_lst.scrollToTop()
	
	def update_captions(self, data):
		# data = {sample_id: [label, value, order], ...}
		
		for sample in self.samples:
			label, value, order = data[sample.id]
			sample.row = order
			sample.value = value
			sample.label = label
		self.samples = sorted(self.samples, key = lambda sample: sample.row)
		
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

