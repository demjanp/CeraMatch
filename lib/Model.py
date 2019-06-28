from deposit import (Store, Commander, Broadcasts)

from lib.fnc_matching import *
from lib.fnc_drawing import *
from lib.Sample import Sample

from PySide2 import (QtCore, QtGui)

from natsort import natsorted
from itertools import product
from pathlib import Path
import numpy as np
import json
import os

FSAMPLE_IDS = "matching_find_ids.json"
FMATCHING = "matching.npy"
DB_URL = "db_typology\\kap_typology.json"

class Model(Store):
	
	def __init__(self, view):
		
		self.view = view
		self.dc = None
		self.samples = [] # [Sample, ...]
		self.sample_ids = []
		self.distance = None
		self.distmax_ordering = {} # {row: sample_id, ...}
		self.weights = {
			"Radius": 0,
			"Tangent": 0,
			"Curvature": 0,
			"Hamming": 0.0,
			"Diameter": 0.20,
			"Axis": 0.80,
		}
#		for name in self.weights:
#			self.weights[name] = 1/len(self.weights)
		self.cluster_weights = {} # {name: weights, ...}
		
		Store.__init__(self, parent = self.view)
		
		root_path = os.path.join(str(Path.home()), "AppData", "Local", "CeraMatch")
		
		self.load(os.path.join(root_path, DB_URL))

		with open(os.path.join(root_path, FSAMPLE_IDS), "r") as f:
			self.sample_ids = json.load(f)
		
		row = 0
		for id in self.classes["Sample"].objects:
			obj = self.objects[id]
			sample_id = obj.descriptors["Id"].label.value
			if sample_id not in self.sample_ids:
				continue
			self.samples.append(Sample(sample_id, obj.descriptors["Reconstruction"].label, sample_id, sample_id, row))
			row += 1
		
		self.distance = np.load(os.path.join(root_path, FMATCHING)) # distance[i, j] = [R_dist, th_dist, kap_dist, h_dist, diam_dist, axis_dist]
		
	def set_weight(self, name, value):
		
		if name not in self.weights:
			return
		self.distmax_ordering = {}
		self.weights[name] = value
		names2 = [name2 for name2 in self.weights if name2 != name]
		values = np.array([self.weights[name2] for name2 in names2])
		sum_tgt = 1 - value
		if values.sum() > 0:
			values *= sum_tgt / values.sum()
		for idx, name2 in enumerate(names2):
			self.weights[name2] = float(values[idx])
			self.view.weights_frame.set_slider(name2, int(round(self.weights[name2] * 100)))
	
	def set_weights(self, weights):
		
		self.weights.update(weights)
		self.view.weights_frame.reload()
	
	def get_distance(self):
		
		return combine_dists(self.distance, self.weights["Radius"], self.weights["Tangent"], self.weights["Curvature"], self.weights["Hamming"], self.weights["Diameter"], self.weights["Axis"])
	
	def get_pca(self, D = None):
		
		if D is None:
			D = self.get_distance()
		if D.shape[0] < 2:
			return None
		pca = PCA(n_components = None)
		pca.fit(D)
		n_components = np.where(np.cumsum(pca.explained_variance_ratio_) > 0.99)[0]
		if not n_components.size:
			return None
		n_components = n_components.min()
		pca = PCA(n_components = n_components)
		pca.fit(D)
		return pca.transform(D)
	
	def get_samples_clusters(self, samples = []):
		
		if not samples:
			samples = self.samples
		clusters = set()
		for sample in samples:
			if sample.cluster is None:
				continue
			clusters.add(sample.cluster)
		return list(clusters)
	
	def load_ids(self):
		
		sorted_ids = natsorted(self.sample_ids)
		for sample in self.samples:
			sample.row = sorted_ids.index(sample.id)
			sample.value = sample.id
			sample.label = sample.id
		self.samples = sorted(self.samples, key = lambda sample: sample.row)
		self.view.image_view.reload()
		self.view.image_view.scrollToTop()
	
	def load_distance(self, sample_id):
		
		D = self.get_distance()
		idx0 = self.sample_ids.index(sample_id)
		idxs = np.argsort(D[idx0]).tolist()
		for sample in self.samples:
			idx1 = self.sample_ids.index(sample.id)
			sample.row = idxs.index(idx1)
			d = D[idx0,idx1]
			sample.value = d
			sample.label = str(round(d, 2))
		self.samples = sorted(self.samples, key = lambda sample: sample.row)
		self.view.image_view.reload()
		self.view.image_view.scrollToTop()
	
	def optimize_weights_all(self, sample_ids, steps = 20):
		
		idxs = [self.sample_ids.index(sample_id) for sample_id in sample_ids]
		distance = self.distance[:,idxs][idxs]
		weights = set() # [[w_R, w_th, w_kap, w_h, w_diam, w_axis], ...]
		values = np.linspace(0, 1, steps).tolist()
		for row in product(*[values for i in range(5)]):
			w_axis = 1 - sum(row)
			if w_axis < 0:
				continue
			weights.add(tuple(list(row) + [w_axis]))
		weights = list(weights)
		test = []
		for w_R, w_th, w_kap, w_h, w_diam, w_axis in weights:
			test.append(combine_dists(distance, w_R, w_th, w_kap, w_h, w_diam, w_axis).mean())
		w_R, w_th, w_kap, w_h, w_diam, w_axis = weights[np.argmax(test)]
		self.set_weights(dict(
			Radius = w_R,
			Tangent = w_th,
			Curvature = w_kap,
			Hamming = w_h,
			Diameter = w_diam,
			Axis = w_axis,
		))
	
	def optimize_weights(self, sample_ids, steps = 20):
		
		idxs = [self.sample_ids.index(sample_id) for sample_id in sample_ids]
		distance = self.distance[:,idxs][idxs]
		weights = set() # [[w_h, w_diam, w_axis], ...]
		values = np.linspace(0, 1, steps + 2)[1:-1].tolist()
		for row in product(*[values for i in range(2)]):
			w_axis = 1 - sum(row)
			if w_axis < values[0]:
				continue
			weights.add(tuple(list(row) + [w_axis]))
		weights = list(weights)
		test = []
		for w_h, w_diam, w_axis in weights:
			test.append(combine_dists(distance, 0, 0, 0, w_h, w_diam, w_axis).std())
		test = np.array(test)
		w_h, w_diam, w_axis = weights[np.argmax(test)]
		self.set_weights(dict(
			Radius = 0,
			Tangent = 0,
			Curvature = 0,
			Hamming = w_h,
			Diameter = w_diam,
			Axis = w_axis,
		))
	
	def populate_distmax_ordering(self):
		
		D = self.get_distance()
		pca_scores = self.get_pca(D)
		if pca_scores is None:
			return
		ordering = get_distmax_ordering(pca_scores, D)
		ordering = dict([(self.sample_ids[idx], ordering[idx]) for idx in ordering])  # {sample_id: row, ...}
		self.distmax_ordering = {}  # {row: sample_id, ...}
		for sample_id in ordering:
			self.distmax_ordering[ordering[sample_id]] = sample_id
		return ordering

	def load_distmax(self):
		
		if self.distmax_ordering:
			ordering = {}  # {sample_id: row, ...}
			for row in self.distmax_ordering:
				ordering[self.distmax_ordering[row]] = row
		else:
			ordering = self.populate_distmax_ordering()
		
		# ordering = {sample_id: order, ...}
		for sample in self.samples:
			order = ordering[sample.id]
			sample.row = order
			sample.value = sample.id
			sample.label = sample.id
		self.samples = sorted(self.samples, key = lambda sample: sample.row)
		self.view.image_view.reload()
		self.view.image_view.scrollToTop()
	
	def browse_distmax(self, direction):
		
		if not self.distmax_ordering:
			self.populate_distmax_ordering()
		# self.distmax_ordering = {row: sample_id, ...}
		
		selected = self.view.get_selected()
		if selected:
			sample_id = selected[0].id
			direction = 0
		else:
			sample_id = self.samples[0].id
			if isinstance(self.samples[0].value, str):
				direction = 0
		for row in self.distmax_ordering:
			if self.distmax_ordering[row] == sample_id:
				break
		if direction == -1:
			if row > 0:
				row -= 1
		elif direction == 1:
			if row < len(self.sample_ids) - 1:
				row += 1
		sample_id = self.distmax_ordering[row]
		self.load_distance(sample_id)
	
	def split_cluster(self, supercluster = None):
		
		if supercluster:
			sample_idxs = [self.sample_ids.index(sample.id) for sample in self.samples if sample.cluster == supercluster]
		else:
			sample_idxs = [self.sample_ids.index(sample.id) for sample in self.samples]
		
		if len(sample_idxs) < 3:
			return
		
		D = self.get_distance()[:,sample_idxs][sample_idxs]
		pca_scores = self.get_pca(D)
		if pca_scores is None:
			return
		
		sample_labels = get_sample_labels(pca_scores)
		sample_labels = dict([(self.sample_ids[sample_idxs[idx]], sample_labels[idx]) for idx in sample_labels])  # {sample_id: label, ...}
		labels_new = set()
		for sample_id in sample_labels:
			labels_new.add(".".join(sample_labels[sample_id].split(".")[:2]))
		label1 = sorted(list(labels_new))[0]
		clusters = {}  # {sample_id: cluster, ...}
		labels = {}  # {sample_id: label, ...}
		for idx in sample_idxs:
			sample_id = self.sample_ids[idx]
			i = 1 if sample_labels[sample_id].startswith(label1) else 2
			if supercluster:
				clusters[sample_id] = "%s.%d" % (supercluster, i)
				labels[sample_id] = "%s.%s" % (supercluster, sample_labels[sample_id])
			else:
				clusters[sample_id] = str(i)
				labels[sample_id] = sample_labels[sample_id]
		
		for sample in self.samples:
			if sample.id in clusters:
				sample.cluster = clusters[sample.id]
				sample.leaf = labels[sample.id]
		
		clusters = set(clusters.values())
		for cluster in clusters:
			self.cluster_weights[cluster] = {}
			for name in self.weights:
				self.cluster_weights[cluster][name] = self.weights[name]
	
	def split_all_clusters(self):
		
		clusters = self.get_samples_clusters()
		if not clusters:
			self.split_cluster(None)
		else:
			for cluster in clusters:
				self.split_cluster(cluster)
	
	def clear_clusters(self):
		
		for sample in self.samples:
			sample.cluster = None
			sample.leaf = None
	
	def join_cluster(self, cluster):
		
		if not cluster:
			return
		
		if "." not in cluster:
			for sample in self.samples:
				sample.cluster = None
				sample.leaf = None
			self.load_ids()
			return
		
		supercluster = cluster.split(".")[:-1]
		cluster = ".".join(supercluster)
		for sample in self.samples:
			if sample.cluster and (sample.cluster.split(".")[:-1] == supercluster):
				sample.cluster = cluster
	
	def manual_cluster(self, samples):
		
		manual_n = 0
		for sample in self.samples:
			if sample.cluster is None:
				continue
			cluster = sample.cluster.split(".")
			for val in cluster:
				if val.startswith("manual_"):
					manual_n = max(manual_n, int(val.split("_")[1]))
		manual_n += 1
		
		clusters = self.get_samples_clusters(samples)
		label = "manual_%d" % (manual_n)
		if clusters:
			supercluster = clusters[0].split(".")
			for cluster in clusters:
				overlap = []
				for i, val in enumerate(cluster.split(".")):
					if (i < len(supercluster)) and (val == supercluster[i]):
						overlap.append(val)
					else:
						break
				if len(overlap) < len(supercluster):
					supercluster = overlap.copy()
				if not supercluster:
					break
			if supercluster:
				label = "%s.manual_%d" % (".".join(supercluster), manual_n)
		
		for sample in samples:
			sample.cluster = label
			sample.leaf = None
		
		new_clusters = [label]
		
		label = "manual_%d" % (manual_n + 1)
		new_clusters.append(label)
		for sample in self.samples:
			if sample.cluster is None:
				sample.cluster = label
		
		for cluster in new_clusters:
			self.cluster_weights[cluster] = {
				"Radius": 0,
				"Tangent": 0,
				"Curvature": 0,
				"Hamming": 1/3,
				"Diameter": 1/3,
				"Axis": 1/3,
			}
	
	def update_leaves(self):
		
		D = self.get_distance()
		clusters = defaultdict(list)  # {cluster: [sample_idx, ...], ...}
		for sample in self.samples:
			if sample.cluster:
				clusters[sample.cluster].append(self.sample_ids.index(sample.id))
		if not clusters:
			pca_scores = self.get_pca(D)
			if pca_scores is None:
				return
			sample_labels = get_sample_labels(pca_scores)
			sample_labels = dict([(self.sample_ids[idx], sample_labels[idx]) for idx in sample_labels])  # {sample_id: label, ...}
		else:
			sample_labels = {}
			for cluster in clusters:
				pca_scores = self.get_pca(D[:,clusters[cluster]][clusters[cluster]])
				if pca_scores is None:
					clu_sample_labels = dict([(idx, label) for idx, label in enumerate(["1"] * len(clusters[cluster]))])
				else:
					clu_sample_labels = get_sample_labels(pca_scores)
				for idx in clu_sample_labels:
					sample_labels[self.sample_ids[clusters[cluster][idx]]] = clu_sample_labels[idx]
		for sample in self.samples:
			sample.value = sample_labels[sample.id]
			sample.leaf = sample_labels[sample.id]
	
	def sort_by_leaf(self):
		
		self.update_leaves()
		self.samples = natsorted(self.samples, key = lambda sample: sample.leaf)
		for row in range(len(self.samples)):
			self.samples[row].row = row
		self.view.image_view.reload()
	
	def set_outlier(self, samples):
		
		for sample in samples:
			sample.outlier = not sample.outlier
			if sample.outlier:
				sample.central = False
	
	def set_central(self, samples):
		
		for sample in samples:
			sample.central = not sample.central
			if sample.central:
				sample.outlier = False
	
	def populate_clusters(self, selected_sample = None):
		
		self.update_leaves()
		cluster_last = None
		ci = True
		self.samples = natsorted(self.samples, key = lambda sample: [sample.cluster, sample.leaf])
		levels_all = set([])
		for row in range(len(self.samples)):
			cluster = self.samples[row].cluster
			if not cluster:
				return
			self.samples[row].row = row
			self.samples[row].value = self.samples[row].leaf
			levels = list(range(1, len(cluster.split(".")) + 1))
			levels_all.update(set(levels))
			self.samples[row].label = dict([(level, None) for level in levels])  # {clustering_level: color, ...}
		
		colors = [QtGui.QColor(253, 231, 37, 255), QtGui.QColor(68, 1, 84, 255)]
		levels_all = natsorted(list(levels_all))
		for level in levels_all:
			cluster_last = None
			ci = True
			for row in range(len(self.samples)):
				if not isinstance(self.samples[row].label, dict):
					continue
				if level not in self.samples[row].label:
					continue
				cluster = ".".join(self.samples[row].cluster.split(".")[:level])
				if cluster != cluster_last:
					ci = not ci
					cluster_last = cluster
				self.samples[row].label[level] = colors[ci]
		
		self.view.image_view.reload()
		if selected_sample is not None:
			self.view.image_view.scrollTo(selected_sample.index)
			self.view.image_view.set_selected([selected_sample.id])
	
	def has_clusters(self):
		
		for sample in self.samples:
			if sample.has_cluster():
				return True
		return False
	
	def max_clustering_level(self):
		
		level = 1
		for sample in self.samples:
			if sample.cluster:
				level = max(level, len(sample.cluster.split(".")))
		return level
	
	def set_datasource(self, data_source):
		
		Store.set_datasource(self, data_source)
	
	def get_temp_dir(self):
		
		tempdir = os.path.normpath(os.path.abspath(os.path.join(tempfile.gettempdir(), "lap")))
		if not os.path.exists(tempdir):
			os.makedirs(tempdir)
		return tempdir
	
	def launch_commander(self):
		
		self.dc = Commander(model = self)
	
	def save_clusters(self, path):
		
		if not self.has_clusters():
			return
		
		# self.cluster_weights = {name: weights, ...}
		data = {} # {cluster: {"weights": weights, "samples": [sample.todict(), ...], ...}
		for cluster in self.cluster_weights:
			data[cluster] = dict(
				weights = dict(self.cluster_weights[cluster].items()),
				samples = [],
			)
		for sample in self.samples:
			if sample.cluster:
				data[sample.cluster]["samples"].append(sample.to_dict())
		with open(path, "w") as f:
			json.dump(data, f)
	
	def load_clusters(self, path):
		
		if not os.path.isfile(path):
			return
		with open(path, "r") as f:
			data = json.load(f)  # {cluster: {"weights": weights, "samples": [sample.todict(), ...], ...}
		self.cluster_weights = {}
		self.sample_data = {} # {sample_id: sample_dict, ...}
		for cluster in data:
			self.cluster_weights[cluster] = {}
			self.cluster_weights[cluster].update(data[cluster]["weights"])
			for sample_dict in data[cluster]["samples"]:
				self.sample_data[sample_dict["id"]] = sample_dict
		for sample in self.samples:
			if sample.id in self.sample_data:
				sample.from_dict(self.sample_data[sample.id])
		self.populate_clusters()
	
	def save_clusters_pdf(self, path):
		
		save_clusters_as_pdf(path, self.samples)
	
	def on_selected(self):
		
		pass

