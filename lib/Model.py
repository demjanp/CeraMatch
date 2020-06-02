from deposit import (Store, Commander, Broadcasts)
from deposit.store.DLabel.DResource import DResource
from deposit.store.DLabel.DString import DString
from deposit.store.DLabel.DGeometry import DGeometry
from deposit import Broadcasts

from lib.fnc_matching import *
from lib.fnc_drawing import *
from lib.Sample import Sample

from PySide2 import (QtCore, QtGui)

from itertools import combinations
from natsort import natsorted
import numpy as np

class Model(Store):
	
	def __init__(self, view):
		
		self.view = view
		self.dc = None
		self.samples = [] # [Sample, ...]
		self.sample_ids = []
		self.distance = None  # distance[i, j] = [diam_dist, axis_dist, h_dist]; i, j: indexes in samples
		self.distance_mean = None  # distance_mean[i, j] = mean dist
		self.distmax_ordering = {} # {row: sample_id, ...}
		self.cluster_history = [] # [data, ...]; data = {cluster: [sample.todict(), ...], ...}
		self.auto_clustering = {} # {sample_id: cluster, ...}
		
		self._last_changed = -1
		
		Store.__init__(self, parent = self.view)
	
	def is_connected(self):
		
		return (self.data_source is not None) and (self.data_source.identifier is not None)
	
	def is_saved(self):
		
		return self._last_changed == self.changed
	
	def has_clusters(self):
		
		if not self.samples:
			return False
		for sample in self.samples:
			if not sample.has_cluster():
				return False
		return True
	
	def has_distances(self):
		
		if self.distance is None:
			return False
		if (self.distance == np.inf).any():
			return False
		return True
	
	def max_clustering_level(self):
		
		level = 1
		for sample in self.samples:
			if sample.cluster:
				level = max(level, len(sample.cluster.split(".")))
		return level
	
	def set_datasource(self, data_source):
		
		Store.set_datasource(self, data_source)
		self.view.on_set_datasource()
	
	def get_sample_by_id(self, sample_id):
		
		for sample in self.samples:
			if sample.id == sample_id:
				return sample
		return None
	
	def get_leaf_labels(self, sample_idxs):
		
		if len(sample_idxs) < 3:
			return None
		D = self.distance_mean[:,sample_idxs][sample_idxs]
		pca_scores = calc_pca_scores(D)
		if pca_scores is None:
			return None
		return get_hca_labels(pca_scores)
	
	
	# Calculate Distances
	
	def calc_distances(self):
		
		if not self.samples:
			return
		profiles = {}  # profiles[sample_id] = [profile, radius]
		for sample in self.samples:
			obj = self.objects[sample.obj_id]
			radius = float(obj.descriptors["Radius"].label.value)
			profile = np.array(obj.descriptors["Profile"].label.coords[0])
			profile = profile - get_rim(profile)
			profile = set_idx0_to_point(profile, np.array([0,0]))
			idx = np.argmax((profile**2).sum(axis = 1))
			profile = np.vstack((profile[idx:], profile[:idx]))
			profile = smoothen_coords(profile)
			profiles[sample.id] = [profile, radius]
		self.distance = calc_distances(profiles, self.distance)
		self.calc_mean_distances()
		
		for i, j in combinations(range(len(self.samples)), 2):
			obj_id1 = self.samples[i].obj_id
			obj_id2 = self.samples[j].obj_id
			diam_dist, axis_dist, h_dist = self.distance[i,j]
			self.objects[obj_id1].relations.add("diam_dist", obj_id2, diam_dist)
			self.objects[obj_id2].relations.add("diam_dist", obj_id1, diam_dist)
			self.objects[obj_id1].relations.add("axis_dist", obj_id2, axis_dist)
			self.objects[obj_id2].relations.add("axis_dist", obj_id1, axis_dist)
			self.objects[obj_id1].relations.add("h_dist", obj_id2, h_dist)
			self.objects[obj_id2].relations.add("h_dist", obj_id1, h_dist)
		
		self.view.update()
	
	def calc_mean_distances(self):
		
		if self.distance is None:
			return
		if (self.distance == np.inf).any():
			return
		mask = (self.distance == 0)
		mins = []
		for idx in range(self.distance.shape[2]):
			d = self.distance[:,:,idx]
			mins.append(d[d > 0].min())
		self.distance_mean = self.distance - mins
		self.distance_mean[mask] = 0
		self.distance_mean = self.distance_mean / self.distance_mean.mean(axis = (0,1))
		self.distance_mean = self.distance_mean.mean(axis = 2)
	
	
	# Load samples
	
	def load_samples(self):
		
		self.samples = []
		self.sample_ids = []
		self.distmax_ordering = {}
		self.cluster_history = []
		self.auto_clustering = {}
		
		descriptors = self.view.descriptor_group.get_values()
		if descriptors:
			cls_sample, descr_id, descr_profile, descr_radius, descr_recons = descriptors
			for obj_id in self.classes[cls_sample].objects:
				obj = self.objects[obj_id]
				sample_id = obj.descriptors[descr_id].label
				profile = obj.descriptors[descr_profile].label
				radius = obj.descriptors[descr_radius].label
				recons = obj.descriptors[descr_recons].label
				if isinstance(sample_id, DString) and isinstance(profile, DGeometry) and isinstance(radius, DString) and isinstance(recons, DResource):
					sample_id = str(sample_id.value)
					self.sample_ids.append(sample_id)
					self.samples.append(Sample(sample_id, recons, sample_id, sample_id, len(self.samples), obj_id))
		
		self.distance = None
		self.distance_mean = None
		if self.samples:
			self.distance = np.zeros((len(self.samples), len(self.samples), 3))
			self.distance[:] = np.inf
			for i, j in combinations(range(len(self.samples)), 2):
				obj_id1 = self.samples[i].obj_id
				obj_id2 = self.samples[j].obj_id
				diam_dist = self.objects[obj_id1].relations["diam_dist"].weight(obj_id2)
				axis_dist = self.objects[obj_id1].relations["axis_dist"].weight(obj_id2)
				h_dist = self.objects[obj_id1].relations["h_dist"].weight(obj_id2)
				if None not in [diam_dist, axis_dist, h_dist]:
					self.distance[i,j] = [float(diam_dist), float(axis_dist), float(h_dist)]
					self.distance[j,i] = [float(diam_dist), float(axis_dist), float(h_dist)]
			for i in range(len(self.samples)):
				self.distance[i,i] = 0
			for sample in self.samples:
				for obj_id2 in self.objects[sample.obj_id].relations["~contains"]:
					obj2 = self.objects[obj_id2]
					if "CMCluster" in obj2.classes:
						cluster = str(obj2.descriptors["Name"].label.value)
						sample.cluster = cluster
			self.calc_mean_distances()
			self.update_leaves()
	
	def sort_samples_by_row(self):
		
		idxs = sorted(range(len(self.samples)), key = lambda idx: self.samples[idx].row)
		self.samples = [self.samples[idx] for idx in idxs]
		self.sample_ids = [self.sample_ids[idx] for idx in idxs]
		if self.distance is not None:
			self.distance = self.distance[:,idxs][idxs]
		if self.distance_mean is not None:
			self.distance_mean = self.distance_mean[:,idxs][idxs]
	
	
	# Sort by Sample IDs
	
	def sort_by_ids(self):
		
		idxs_sort = natsorted(range(len(self.samples)), key = lambda idx: self.samples[idx].id)
		for row, idx in enumerate(idxs_sort):
			self.samples[idx].row = row
			self.samples[idx].value = self.samples[idx].id
			self.samples[idx].label = self.samples[idx].id
		self.sort_samples_by_row()
		self.view.image_view.reload()
		self.view.image_view.scrollToTop()
	
	
	# Sort by Max/Min Distance
	
	def populate_distmax_ordering(self):
		
		ordering = get_distmax_ordering(self.distance_mean)
		ordering = dict([(self.sample_ids[idx], ordering[idx]) for idx in ordering])  # {sample_id: row, ...}
		self.distmax_ordering = {}  # {row: sample_id, ...}
		for sample_id in ordering:
			self.distmax_ordering[ordering[sample_id]] = sample_id
		return ordering

	def sort_by_distmax(self):
		
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
		self.sort_samples_by_row()
		self.view.image_view.reload()
		self.view.image_view.scrollToTop()
	
	def sort_by_distance(self, sample_id):
		
		idx0 = self.sample_ids.index(sample_id)
		idxs = np.argsort(self.distance_mean[idx0]).tolist()
		for sample in self.samples:
			idx1 = self.sample_ids.index(sample.id)
			sample.row = idxs.index(idx1)
			sample.value = self.distance_mean[idx0,idx1]
			sample.label = str(round(self.distance_mean[idx0,idx1], 2))
		self.sort_samples_by_row()
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
		self.sort_by_distance(sample_id)	
	
	
	# Sort by Clustering
	
	def update_leaves(self):
		
		clusters = set([])
		for sample in self.samples:
			if sample.cluster is not None:
				clusters.add(sample.cluster)
			else:
				sample.leaf = None
				sample.value = ""
		for cluster in clusters:
			sample_idxs = [idx for idx in range(len(self.samples)) if self.samples[idx].cluster == cluster]
			sample_labels = self.get_leaf_labels(sample_idxs)
			if sample_labels is None:
				sample_idxs = natsorted(sample_idxs, key = lambda idx: self.samples[idx].id)
			else:
				sample_idxs = natsorted(sample_idxs, key = lambda idx: sample_labels[sample_idxs.index(idx)])
			for i, idx in enumerate(sample_idxs):
				self.samples[idx].leaf = "%s.%d" % (cluster, i + 1)
				self.samples[idx].value = self.samples[idx].leaf
	
	def sort_by_cluster(self):
		
		idxs_sort = natsorted(range(len(self.samples)), key = lambda idx: [self.samples[idx].cluster, self.samples[idx].leaf])
		for row, idx in enumerate(idxs_sort):
			self.samples[idx].row = row
			self.samples[idx].value = self.samples[idx].leaf
		self.sort_samples_by_row()
	
	def color_clusters(self, selected = None):
		
		colors = [QtGui.QColor(253, 231, 37, 255), QtGui.QColor(68, 1, 84, 255)]
		cluster_last = None
		levels_all = set([])
		cluster_last = None
		ci = True
		for row in range(len(self.samples)):
			cluster = self.samples[row].cluster
			if not cluster:
				return
			self.samples[row].row = row
			self.samples[row].value = self.samples[row].leaf
			levels = list(range(1, len(cluster.split(".")) + 1))
			levels_all.update(set(levels))
			if self.samples[row].cluster != cluster_last:
				ci = not ci
				cluster_last = cluster
			self.samples[row].label = {"list": colors[ci], "table": dict([(level, None) for level in levels])}  # {list: color, table: {clustering_level: color, ...}}
		
		ci = True
		levels_all = natsorted(list(levels_all))
		for level in levels_all:
			cluster_last = None
			ci = True
			for row in range(len(self.samples)):
				if not isinstance(self.samples[row].label, dict):
					continue
				if level not in self.samples[row].label["table"]:
					continue
				cluster = ".".join(self.samples[row].cluster.split(".")[:level])
				if cluster != cluster_last:
					ci = not ci
					cluster_last = cluster
				self.samples[row].label["table"][level] = colors[ci]
		
		self.view.image_view.reload()
		if selected is not None:
			self.view.image_view.scrollTo(selected.index)
			self.view.image_view.set_selected([selected.id])
	
	def consolidate_cluster_names(self):
		
		clusters = set([])
		for sample in self.samples:
			if sample.cluster is not None:
				clusters.add(sample.cluster)
		rename = {}  # {cluster: cluster_new, ...}
		for cluster in clusters:
			cluster_updated = cluster
			cluster_new = cluster_updated
			while True:
				cluster_new = ".".join(cluster_updated.split(".")[:-1])
				if not cluster_new:
					break
				if len([clu for clu in clusters if clu.startswith(cluster_new)]) < 2:
					cluster_updated = cluster_new
				else:
					break
			if cluster_updated != cluster:
				rename[cluster] = cluster_updated
		if rename:
			for sample in self.samples:
				if sample.cluster in rename:
					cluster_updated = rename[sample.cluster]
					sample.cluster = cluster_updated
					sample.label = cluster_updated
	
	def update_clusters(self, selected = None):
		
		self.consolidate_cluster_names()
		self.update_leaves()
		self.sort_by_cluster()
		self.color_clusters(selected)
	
	
	# Auto Cluster
	
	def load_auto_clustering(self):
		
		if not self.has_distances():
			return
		if not self.auto_clustering:
			clusters = get_clusters(self.distance) # {label: [i, ...], ...}; i = index in sample_ids
			for label in clusters:
				for idx in clusters[label]:
					self.auto_clustering[self.samples[idx].id] = label
	
	def auto_cluster(self):
		
		self.update_cluster_history()
		self.load_auto_clustering()
		for sample in self.samples:
			sample.cluster = self.auto_clustering[sample.id]
		
		self.update_clusters()
		self.save_clusters_to_db()
	
	
	# Split
	
	def split_cluster(self, supercluster, selected):
		
		sample_idxs = [idx for idx in range(len(self.samples)) if self.samples[idx].cluster == supercluster]
		if len(sample_idxs) < 3:
			return
		sample_labels = self.get_leaf_labels(sample_idxs)
		if sample_labels is None:
			return
		self.update_cluster_history()
		sample_labels = dict([(self.sample_ids[sample_idxs[idx]], label) for idx, label in enumerate(sample_labels)])  # {sample_id: label, ...}
		labels_new = set()
		for sample_id in sample_labels:
			labels_new.add(".".join(sample_labels[sample_id].split(".")[:2]))
		label1 = sorted(list(labels_new))[0]
		clusters = {}  # {sample_id: cluster, ...}
#		labels = {}  # {sample_id: label, ...}
		for idx in sample_idxs:
			sample_id = self.sample_ids[idx]
			i = 1 if sample_labels[sample_id].startswith(label1) else 2
			clusters[sample_id] = "%s.%d" % (supercluster, i)
#			labels[sample_id] = "%s_%s" % (supercluster, sample_labels[sample_id])
		
		for sample in self.samples:
			if sample.id in clusters:
				sample.cluster = clusters[sample.id]
		
		self.update_clusters(selected)
		self.save_clusters_to_db()
	
	
	# Join to Parent
	
	def join_cluster_to_parent(self, cluster, selected):
		
		if not cluster:
			return
		
		self.update_cluster_history()
		
		if "." not in cluster:
			for sample in self.samples:
				if sample.cluster == cluster:
					sample.cluster = None
		else:
			supercluster = ".".join(cluster.split(".")[:-1])
			for sample in self.samples:
				if sample.cluster == cluster:
					sample.cluster = supercluster
		
		self.update_clusters(selected)
		self.save_clusters_to_db()
	
	
	# Join Children
	
	def join_children_to_cluster(self, cluster, level, selected):
		
		if not cluster:
			return
		
		self.update_cluster_history()
		
		cluster = ".".join(cluster.split(".")[:level])
		for sample in self.samples:
			if sample.cluster is None:
				continue
			if sample.cluster.startswith("%s." % (cluster)):
				sample.cluster = cluster
		
		self.update_clusters(selected)
		self.save_clusters_to_db()
	
	
	# Create Manual
	
	def manual_cluster(self, samples, selected):
		
		def get_samples_clusters(samples):
			
			if not samples:
				samples = self.samples
			clusters = set()
			for sample in samples:
				if sample.cluster is None:
					continue
				clusters.add(sample.cluster)
			return list(clusters)
		
		self.update_cluster_history()
		
		manual_n = 0
		for sample in self.samples:
			if sample.cluster is None:
				continue
			cluster = sample.cluster.split(".")
			for val in cluster:
				if val.startswith("manual_"):
					manual_n = max(manual_n, int(val.split("_")[1]))
		manual_n += 1
		
		clusters = get_samples_clusters(samples)
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
		
		label = "manual_%d" % (manual_n + 1)
		for sample in self.samples:
			if sample.cluster is None:
				sample.cluster = label
		
		self.update_clusters(selected)
		self.save_clusters_to_db()
	
	def add_to_cluster(self, src_ids, tgt_id):
		
		if not src_ids:
			return False
		sample_tgt = self.get_sample_by_id(tgt_id)
		if (sample_tgt is None) or (sample_tgt.cluster is None):
			return False
		self.update_cluster_history()
		cluster_tgt = sample_tgt.cluster
		clusters_all = set([])
		for sample in self.samples:
			if sample.cluster is not None:
				clusters_all.add(sample.cluster)
			if sample.id in src_ids:
				sample.cluster = cluster_tgt
				sample.label = cluster_tgt
		self.update_clusters(self.get_sample_by_id(tgt_id))
		self.save_clusters_to_db()
		return True
	
	
	# Clear All
	
	def clear_clusters(self):
		
		for sample in self.samples:
			sample.cluster = None
			sample.leaf = None
			sample.value = None
		self.save_clusters_to_db()
	
	
	# History
	
	def has_history(self):
		
		if self.cluster_history:
			return True
		return False
	
	def clusters_todict(self):
		
		data = {} # {sample_id: cluster, ...}
		
		if not self.has_clusters():
			return data
		for sample in self.samples:
			data[sample.id] = sample.cluster
		return data
	
	def clusters_fromdict(self, data):
		
		for sample in self.samples:
			if sample.id in data:
				sample.cluster = data[sample.id]
	
	def update_cluster_history(self):
		
		data = self.clusters_todict()
		if data:
			self.cluster_history.append(data)
			self.view.update()
	
	def undo_clustering(self):
		
		if not self.cluster_history:
			return
		data = self.cluster_history.pop()
		if not data:
			return
		self.clusters_fromdict(data)
		self.update_clusters()
		self.save_clusters_to_db()
	
	
	# Launch Deposit
	
	def launch_deposit(self):
		
		self.dc = Commander(model = self)
	
	
	# Save
	
	def save_clusters_to_db(self):
		
		self.classes.add("CMCluster")
		for obj_id in list(self.classes["CMCluster"].objects.keys()):
			del self.objects[obj_id]
		cluster_ids = {} # {name: obj_id, ...}
		for sample in self.samples:
			if sample.cluster is not None:
				if sample.cluster in cluster_ids:
					clu_id = cluster_ids[sample.cluster]
				else:
					clu_id = self.classes["CMCluster"].add_object()
					clu_id.add_descriptor("Name", sample.cluster)
					clu_id = clu_id.id
					cluster_ids[sample.cluster] = clu_id
				self.objects[clu_id].relations.add("contains", sample.obj_id)
	
	def save_clusters_pdf(self, path):
		
		save_clusters_as_pdf(path, self.samples)
	
	
	# Interface events
	
	def on_selected(self):
		
		pass
	
	def on_close(self):
		
		if self.dc is not None:
			self.dc.close()


