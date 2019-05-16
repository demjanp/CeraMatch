from deposit import (Store, Commander, Broadcasts)

from lib.fnc_matching import *
from lib.Sample import Sample

from PySide2 import (QtCore, QtGui)
from natsort import (natsorted)
import numpy as np
import json

FSAMPLE_IDS = "data/matching_find_ids.json"
FMATCHING = "data/matching.npy"
DB_URL = "data\\db_typology\\kap_typology.json"

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
			"Hamming": 0.10,
			"Diameter": 0.30,
			"Landmarks": 0.60,
		}
#		for name in self.weights:
#			self.weights[name] = 1/len(self.weights)
		
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
		
		self.distance = np.load(FMATCHING) # distance[i, j] = [R_dist, th_dist, kap_dist, h_dist, diam_dist, land_dist]
		
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
	
	def get_distance(self):
		
		return combine_dists(self.distance, self.weights["Radius"], self.weights["Tangent"], self.weights["Curvature"], self.weights["Hamming"], self.weights["Diameter"], self.weights["Landmarks"])
	
	def get_pca(self, D = None):
		
		if D is None:
			D = self.get_distance()
		pca = PCA(n_components = None)
		pca.fit(D)
		n_components = np.where(np.cumsum(pca.explained_variance_ratio_) > 0.9)[0].min()
		pca = PCA(n_components = n_components)
		pca.fit(D)
		return pca.transform(D)
		
	def load_ids(self):
		
		sorted_ids = natsorted(self.sample_ids)
		for sample in self.samples:
			sample.row = sorted_ids.index(sample.id)
			sample.value = sample.id
			sample.label = sample.id
		self.samples = sorted(self.samples, key = lambda sample: sample.row)
		self.view.image_lst.reload()
		self.view.image_lst.scrollToTop()
		
	def load_distance(self, sample_id):
		
		pca_scores = self.get_pca()
		D = squareform(pdist(pca_scores))
		
		data = {} # {sample_id: [rounded distance, distance, order], ...}
		idx0 = self.sample_ids.index(sample_id)
		idxs = np.argsort(D[idx0]).tolist()
		for sample in self.samples:
			idx1 = self.sample_ids.index(sample.id)
			sample.row = idxs.index(idx1)
			d = D[idx0,idx1]
			sample.value = d
			sample.label = str(round(d, 2))
		self.samples = sorted(self.samples, key = lambda sample: sample.row)
		self.view.image_lst.reload()
		self.view.image_lst.scrollToTop()

	def populate_distmax_ordering(self):
		
		pca_scores = self.get_pca()
		ordering = get_distmax_ordering(pca_scores)
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
		self.view.image_lst.reload()
		self.view.image_lst.scrollToTop()
	
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
	
	def split_cluster(self, selected_sample):
		
		supercluster = selected_sample.cluster
		if supercluster:
			sample_idxs = [self.sample_ids.index(sample.id) for sample in self.samples if sample.cluster == supercluster]
		else:
			sample_idxs = [self.sample_ids.index(sample.id) for sample in self.samples]
		
		D = self.get_distance()[:,sample_idxs][sample_idxs]
		pca_scores = self.get_pca(D)
		
		sample_labels = get_sample_labels(pca_scores)
		sample_labels = dict([(self.sample_ids[sample_idxs[idx]], sample_labels[idx]) for idx in sample_labels])  # {sample_id: label, ...}
		labels_new = set()
		for sample_id in sample_labels:
			labels_new.add(".".join(sample_labels[sample_id].split(".")[:2]))
		label1 = sorted(list(labels_new))[0]
		clusters = {} # {sample_id: cluster, ...}
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
		self.load_clusters(selected_sample)
	
	def join_cluster(self, selected_sample):
		
		cluster = selected_sample.cluster
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
		self.load_clusters(selected_sample)
	
	def update_leaves(self):
		
		D = self.get_distance()
		clusters = defaultdict(list)  # {cluster: [sample_idx, ...], ...}
		for sample in self.samples:
			if sample.cluster:
				clusters[sample.cluster].append(self.sample_ids.index(sample.id))
		if not clusters:
			pca_scores = self.get_pca(D)
			sample_labels = get_sample_labels(pca_scores)
			sample_labels = dict([(self.sample_ids[idx], sample_labels[idx]) for idx in sample_labels])  # {sample_id: label, ...}
		else:
			sample_labels = {}
			for cluster in clusters:
				pca_scores = self.get_pca(D[:,clusters[cluster]][clusters[cluster]])
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
		self.view.image_lst.reload()
	
	def load_clusters(self, selected_sample = None):
		
		self.update_leaves()
		cluster_last = None
		ci = True
		colors = [QtGui.QColor(210, 210, 210, 255), QtGui.QColor(128, 128, 128, 255)]
		self.samples = natsorted(self.samples, key = lambda sample: [sample.cluster, sample.leaf])
		for row in range(len(self.samples)):
			cluster = self.samples[row].cluster
			if not cluster:
				return
			self.samples[row].row = row
			self.samples[row].value = self.samples[row].leaf
			if cluster != cluster_last:
				ci = not ci
				cluster_last = cluster
			self.samples[row].label = colors[int(ci)]
		self.view.image_lst.reload()
		if selected_sample is not None:
			self.view.image_lst.scrollTo(selected_sample.index)
			self.view.image_lst.set_selected([selected_sample.id])
	
	def has_clusters(self):
		
		for sample in self.samples:
			if sample.has_cluster():
				return True
		return False
	
	def set_datasource(self, data_source):
		
		Store.set_datasource(self, data_source)
	
	def get_temp_dir(self):
		
		tempdir = os.path.normpath(os.path.abspath(os.path.join(tempfile.gettempdir(), "lap")))
		if not os.path.exists(tempdir):
			os.makedirs(tempdir)
		return tempdir
	
	def launch_commander(self):
		
		self.dc = Commander(model = self)
	
	def on_selected(self):
		
		pass

