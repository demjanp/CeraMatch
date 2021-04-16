
from lib.fnc_drawing import *
from lib.fnc_matching import *

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
		
		self.sample_ids = []
		self.sample_data = {}
		self.distance = None
		
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
	
	def clear_samples(self):
		
		self.sample_ids = []
		self.sample_data = {}
		self.distance = None
		self._has_distance = False
	
	def load_samples(self, lap_descriptors):
		
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
		
		if lap_descriptors is None:
			return
		
		sample_cls, id_descr = lap_descriptors["Custom_Id"]
		for obj_id in self.classes[sample_cls].objects:
			sample_id, descriptors = load_drawing_data(self, lap_descriptors, obj_id)
			if sample_id is None:
				continue
			if not _check_profile(descriptors):
				continue
			if sample_id in self.sample_ids:
				continue
			self.sample_ids.append(sample_id)
			self.sample_data[sample_id] = [obj_id, descriptors]
		
		n_samples = len(self.sample_ids)
		self.distance = np.full((n_samples, n_samples, 4), np.inf, dtype = float)
		self.distance[np.diag_indices(n_samples)] = 0
		for i, j in combinations(range(n_samples), 2):
			obj_id1 = self.sample_data[self.sample_ids[i]][0]
			obj_id2 = self.sample_data[self.sample_ids[j]][0]
			diam_dist = self.objects[obj_id1].relations["diam_dist"].weight(obj_id2)
			axis_dist = self.objects[obj_id1].relations["axis_dist"].weight(obj_id2)
			h_dist = self.objects[obj_id1].relations["h_dist"].weight(obj_id2)
			h_rim_dist = self.objects[obj_id1].relations["h_rim_dist"].weight(obj_id2)
			if None not in [diam_dist, axis_dist, h_dist, h_rim_dist]:
				self.distance[i, j] = [diam_dist, axis_dist, h_dist, h_rim_dist]
				self.distance[j, i] = self.distance[i, j]
		
		self._has_distance = (not (self.distance == np.inf).any())
		
		# TODO load clustering if available
	
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
			diam_dist, axis_dist, h_dist, h_rim_dist = self.distance[i,j]
			self.objects[obj_id1].relations.add("diam_dist", obj_id2, diam_dist)
			self.objects[obj_id2].relations.add("diam_dist", obj_id1, diam_dist)
			self.objects[obj_id1].relations.add("axis_dist", obj_id2, axis_dist)
			self.objects[obj_id2].relations.add("axis_dist", obj_id1, axis_dist)
			self.objects[obj_id1].relations.add("h_dist", obj_id2, h_dist)
			self.objects[obj_id2].relations.add("h_dist", obj_id1, h_dist)
			self.objects[obj_id1].relations.add("h_rim_dist", obj_id2, h_rim_dist)
			self.objects[obj_id2].relations.add("h_rim_dist", obj_id1, h_rim_dist)
	
	def get_clusters(self, max_clusters, limit):
		
		def _format_id(idx):
			
			if idx < len(self.sample_ids):
				return str(self.sample_ids[idx])
			else:
				return "#%d" % (idx)
		
		if not self._has_distance:
			return None, None, None
		clusters, nodes, edges, labels = get_clusters(self.distance, max_clusters = max_clusters, limit = limit)
		# clusters = {idx: [i, ...], ...}; idx = index in nodes; i = index in sample_ids
		# nodes = [idx, ...]; idx < len(samples) -> index in sample_ids
		# edges = [(idx1, idx2), ...]
		# labels = {idx: label, ...}
		
		edges = set([(_format_id(idx1), _format_id(idx2)) for idx1, idx2 in edges])
		nodes = set([_format_id(idx) for idx in nodes])
		clusters = dict([(_format_id(idx), [_format_id(i) for i in clusters[idx]]) for idx in clusters])
		labels = dict([(_format_id(idx), labels[idx]) for idx in labels])
		
		return nodes, edges, clusters, labels
	
	def launch_deposit(self):
		
		self.dc = Commander(model = self)
	
	def on_close(self):
		
		if self.dc is not None:
			self.dc.close()
