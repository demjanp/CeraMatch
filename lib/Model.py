
from lib.fnc_drawing import *
from lib.fnc_matching import *

from lib.Clusters import (Clusters)

from deposit import (Broadcasts)
from deposit.store.Store import (Store)
from deposit.commander.Commander import (Commander)

from PySide2 import (QtCore, QtGui)
from itertools import combinations
import numpy as np

class Model(Store):
	
	def __init__(self, view):
		
		self.view = view
		self.dc = None
		
		self.clusters = None
		
		self.lap_descriptors = None
		self.cluster_class = None
		self.node_class = None
		
		self.sample_ids = []
		self.sample_data = {}
		self.distance = None
		self.dist_rels = ["diam_dist", "axis_dist", "h_dist", "h_rim_dist"]
		
		# self.sample_data = {sample_id: [obj_id, descriptors], ...}
		#	descriptors = {
		#		profile = {Profile_Geometry, Profile_Rim, Profile_Bottom, Profile_Radius, Profile_Radius_Point, Profile_Rim_Point, Profile_Bottom_Point, Arc_Geometry: []}
		#		details = {target_id: {Detail_Geometry, Detail_Closed, Detail_Filled}, ...}
		#		inflections = {target_id: {Inflection_Geometry, Inflection_Dashed}, ...}
		#		breaks = {target_id: {Break_Geometry}, ...}
		#	}
		
		self._has_distance = False
		self._last_changed = -1
		
		Store.__init__(self, parent = self.view)
		
		self.clusters = Clusters(self)
		
		self.broadcast_timer = QtCore.QTimer()
	
	def set_datasource(self, data_source):
		
		self.clear_samples()
		Store.set_datasource(self, data_source)
	
	def is_connected(self):
		
		return (self.data_source is not None) and (self.data_source.identifier is not None)
	
	def is_saved(self):
		
		return self._last_changed == self.changed
	
	def has_history(self):
		
		return False
	
	def has_samples(self):
		
		return len(self.sample_data) > 0
	
	def has_distance(self):
		
		return self._has_distance
	
	def has_cluster_classes(self):
		
		if self.cluster_class and self.node_class:
			return True
		return False
	
	def has_clusters(self):
		
		return self.clusters.has_clusters()
	
	def load_lap_descriptors(self, descriptors = None, sample_cls = None):
		
		if (descriptors is None) or (self.lap_descriptors is None):
			self.lap_descriptors = get_lap_descriptors(descriptors)
		if not sample_cls:
			self.lap_descriptors = None
		else:
			for name in [
				"Custom_Id",
				"Profile_Rim",
				"Profile_Bottom",
				"Profile_Radius",
				"Profile_Geometry",
				"Profile_Radius_Point",
				"Profile_Rim_Point",
				"Profile_Bottom_Point",
				"Profile_Left_Side",
			]:
				self.lap_descriptors[name][0] = sample_cls
	
	def clear_samples(self):
		
		self.clusters.clear()
		
		self.sample_ids = []
		self.sample_data = {}
		self.distance = None
		self._has_distance = False
	
	def calc_distance(self):
		
		if not self.has_samples():
			return
		profiles = {}  # profiles[sample_id] = [profile, radius]
		for sample_id in self.sample_data:
			descriptors = self.sample_data[sample_id][1]
			profile = np.array(descriptors["profile"]["Profile_Geometry"].coords[0])
			radius = float(descriptors["profile"]["Profile_Radius"].value)
			profile = get_reduced(profile, 0.5)
			profile = get_interpolated(profile, 0.5)
			profiles[sample_id] = [profile, radius]
		self.distance = calc_distances(profiles, self.distance)
		
		self._has_distance = (not (self.distance == np.inf).any())
		
		for i, j in combinations(range(len(self.sample_ids)), 2):
			obj_id1 = self.sample_data[self.sample_ids[i]][0]
			obj_id2 = self.sample_data[self.sample_ids[j]][0]
			for k, name in enumerate(self.dist_rels):
				d = self.distance[i,j,k]
				self.objects[obj_id1].relations.add(name, obj_id2, d)
				self.objects[obj_id2].relations.add(name, obj_id1, d)
	
	def delete_distance(self):
		
		for sample_id in self.sample_data:
			obj = self.objects[self.sample_data[sample_id][0]]
			for rel in self.dist_rels:
				obj.del_relation(rel)
		self._has_distance = False
	
	def load_distance(self):
		
		if not self.sample_ids:
			self._has_distance = False
			self.distance = None
			return
		
		n_samples = len(self.sample_ids)
		self.distance = np.full((n_samples, n_samples, 4), np.inf, dtype = float)
		self.distance[np.diag_indices(n_samples)] = 0
		obj_lookup = {} # {obj_id: sample_idx, ...}
		for i, sample_id in enumerate(self.sample_ids):
			obj_lookup[self.sample_data[sample_id][0]] = i
		for i, sample_id in enumerate(self.sample_ids):
			obj = self.objects[self.sample_data[sample_id][0]]
			for k, name in enumerate(self.dist_rels):
				if name not in obj.relations:
					continue
				rel = obj.relations[name]
				for target_id in rel._weights:
					d = rel._weights[target_id]
					if (d is not None) and (target_id in obj_lookup):
						j = obj_lookup[target_id]
						self.distance[i,j,k] = d
						self.distance[j,i,k] = d
		
		self._has_distance = (not (self.distance == np.inf).any())
	
	def load_samples(self):
		# returns clusters, nodes, edges, labels
		#
		# clusters = {#node_id: [@sample_id, ...], ...}
		# nodes = [@sample_id, #node_id, ...]
		# edges = [(@sample_id, #node_id), (#node_id1, #node_id2), ...]
		# labels = {@sample_id: label, #node_id: label, ...}
		
		def _check_profile(descriptors):
			
			if "Profile_Geometry" not in descriptors["profile"]:
				return False
			if "Profile_Radius" not in descriptors["profile"]:
				return False
			
			radius = descriptors["profile"]["Profile_Radius"].value
			try:
				radius = float(radius)
			except:
				return False
			if radius <= 0:
				return False
			if descriptors["profile"]["Profile_Geometry"].__class__.__name__ != "DGeometry":
				return False
			if len(descriptors["profile"]["Profile_Geometry"].coords[0]) < 5:
				return False
			return True
		
		self.clear_samples()
		
		if self.lap_descriptors is None:
			return None, None, None, None
		
		sample_cls, id_descr = self.lap_descriptors["Custom_Id"]
		for obj_id in self.classes[sample_cls].objects:
			sample_id, descriptors = load_drawing_data(self, self.lap_descriptors, obj_id)
			if sample_id is None:
				continue
			if not _check_profile(descriptors):
				continue
			if sample_id in self.sample_ids:
				continue
			obj = self.objects[obj_id]
			self.sample_ids.append(sample_id)
			self.sample_data[sample_id] = [obj_id, descriptors]
		
		self.clusters.update_samples()
		
		self.load_distance()
		
		return self.clusters.load()
	
	def launch_deposit(self):
		
		self.dc = Commander(model = self)
	
	def on_close(self):
		
		if self.dc is not None:
			self.dc.close()


