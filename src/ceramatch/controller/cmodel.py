from lap_data import LCModel

from deposit_gui import DRegistry

from deposit.datasource import (AbstractDatasource, Memory)
from deposit import (DDateTime, DGeometry, DResource)
from deposit.utils.fnc_files import (as_url)
from deposit.query.parse import (remove_bracketed_all)

from PySide6 import (QtWidgets, QtCore, QtGui)
from collections import defaultdict
from itertools import combinations
import datetime
import winreg
import json
import os


class CModel(LCModel):
	
	def __init__(self, cmain):
		
		LCModel.__init__(self, cmain)
		
		self._primary_class = None
		self._cm_classes = {
			"cluster": "CMCluster", 
			"node": "CMNode",
			"position": "CMPosition",
		}
		self._dist_rels = [
			"diam_dist", "axis_dist", "dice_dist", "dice_rim_dist"
		]
		self._distance = {}
		self._query = self.get_query("")
		self._objects = None
	
	
	# ---- Signal handling
	# ------------------------------------------------------------------------
	def on_set_descriptors(self, data):
		# data = [[name, chain], ...];
		# 	chain = "Class.Descriptor" or "Class.Relation.Class.Descriptor"
		
		pass
		
	def on_set_attributes(self, data):
		# data = [(label, ctrl_type, name), ...]
		
		self.cmain.ccontrols.set_attributes(data)
	
	def on_added(self, objects, classes):
		# elements = [DObject, DClass, ...]
		
		self.cmain.cactions.update()
		self.cmain.ccontrols.update()
	
	def on_deleted(self, objects, classes):
		# elements = [obj_id, name, ...]
		
		self.cmain.cactions.update()
		self.cmain.ccontrols.update()
	
	def on_changed(self, objects, classes):
		# elements = [DObject, DClass, ...]
		
		self.cmain.cactions.update()
		self.cmain.ccontrols.update()
	
	def on_saved(self, datasource):
		
		self.cmain.cview.set_status_message("Saved: %s" % (str(datasource)))
		self.cmain.cactions.update()
	
	def on_loaded(self):
		
		self.cmain.clear()
		self.update_model_info()
		self.cmain.ccontrols.on_loaded()
		self.cmain.cview.on_loaded()
		self.cmain.cactions.update()
	
	def on_settings_changed(self):
		
		self.cmain.cactions.update()
		self.cmain.ccontrols.update()
	
	
	# ---- get/set
	# ------------------------------------------------------------------------
	def clear(self):
		
		self._distance = {}
		self._query = self.get_query("")
		self._objects = None
	
	def set_cm_classes(self, data):
		# data = {"cluster": name, "node": name, "position": name}
		
		self._cm_classes = data
	
	def load_descriptors(self):
		
		def _get_data(lap_registry, name):
			
			data = lap_registry.get(name)
			if not data:
				return []
			return json.loads(data)
		
		lap_registry = DRegistry("Laser Aided Profiler")
		self.set_descriptors(_get_data(lap_registry, "descriptors"))
		self.set_attributes(_get_data(lap_registry, "attributes"))
		self.set_cm_classes(
			self.cmain.cview.get_registry("cm_classes", self._cm_classes)
		)
	
	def get_primary_class(self):
		# returns class of Sample
		
		if self._primary_class is None:
			for name, chain in self._descriptors:
				if name == self.NAME_ID:
					self._primary_class = self.parse_chain(chain)[0]
					break
		
		return self._primary_class
	
	def get_cm_classes(self):
		# returns {"cluster": name, "node": name, "position": name}
		
		return self._cm_classes
	
	def set_query(self, querystr):
		
		self._distance = {}
		self._objects = None
		self._query.querystr = querystr
		self._query.process()
	
	def get_drawing_objects(self):
		
		if self._objects is None:
			self._objects = []
			for row in self._query:
				obj = self.get_object(row[0][0])
				if obj is not None:
					self._objects.append(obj)
		return self._objects
	
	def load_drawings(self):
		
		def _get_structure():
			
			_, descriptors, _ = self.get_data_structure()
			name_lookup = {}  # {(Class, Descriptor): name, ...}
			primary_class = None # name of Sample class
			for name, cls, descr in descriptors:
				name_lookup[(cls, descr)] = name
				if name == self.NAME_ID:
					primary_class = cls
			if primary_class is None:
				raise Exception("Sample Class not found")
			return name_lookup, primary_class
			
		
		
		def _get_position(obj, descr_pos):
			
			pos = obj.get_descriptor(descr_pos)
			if isinstance(pos, DGeometry):
				x, y = pos.coords
				return (x, y)
			return None
		
		name_lookup, primary_class = _get_structure()
		
		objects = self.get_drawing_objects()
		if not objects:
			return
		for obj in objects:
			data = self.load_object_data(obj.id, name_lookup, primary_class).get("Settings", None)
			if isinstance(data, dict) and ('descriptors' in data):
				settings = {"descriptors": [], "attributes": []}
				if 'descriptors' in data['descriptors']:
					settings['descriptors'] = data['descriptors']['descriptors']
				if 'attributes' in data['descriptors']:
					settings['attributes'] = data['descriptors']['attributes']
				if settings['descriptors']:
					self.from_settings_dict(settings)
					break
		
		name_lookup, primary_class = _get_structure()
		
		recons_descr = None
		for cls, descr in name_lookup:
			if name_lookup[(cls, descr)] == "Reconstruction":
				recons_descr = descr
				break
		
		drawing_data = {}  
		# {obj_id: {key: value, key: [{key: value, ...}, ...], ...}, ...}
		cluster_data = {}
		# {obj_id: {"name": value, "position": (x, y), "children": [obj_id, ...], ...}, ...}
		node_data = {}
		# {obj_id: {"name": value, "position": (x, y), ...}, ...}
		edges = set()
		# [(source_id, target_id), ...]
		
		descr_pos = self._cm_classes["position"]
		
		QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
		
		self._progress.show("Loading Drawings")
		self._progress.update_state(value = 0, maximum = len(objects))
		cnt = 1
		for obj in objects:
			self._progress.update_state(value = cnt)
			cnt += 1
			if self._progress.cancel_pressed():
				break
			
			drawing_data[obj.id] = self.load_object_data(obj.id, name_lookup, primary_class)
			if self.NAME_ID not in drawing_data[obj.id]:
				del drawing_data[obj.id]
				continue
			pos = _get_position(obj, descr_pos)
			if pos is not None:
				drawing_data[obj.id]["position"] = pos
		
		self._distance = {}
		self._progress.show("Loading Distances")
		self._progress.update_state(value = 0, maximum = len(objects))
		cnt = 1
		done = set()
		for obj in objects:
			self._progress.update_state(value = cnt)
			cnt += 1
			if self._progress.cancel_pressed():
				break
			
			for obj_tgt, label in obj.get_relations():
				if label not in self._dist_rels:
					continue
				w = obj.get_relation_weight(obj_tgt.id, label)
				if w is None:
					continue
				idx = self._dist_rels.index(label)
				if obj.id not in self._distance:
					self._distance[obj.id] = {}
				if obj_tgt.id not in self._distance[obj.id]:
					self._distance[obj.id][obj_tgt.id] = [None, None, None, None]
				self._distance[obj.id][obj_tgt.id][idx] = w
				done.add((obj.id, obj_tgt.id))
		
		for obj_id1, obj_id2 in combinations([obj.id for obj in objects], 2):
			if ((obj_id1, obj_id2) in done) or ((obj_id2, obj_id1) in done):
				continue
			if obj_id1 not in self._distance:
				self._distance[obj_id1] = {}
			self._distance[obj_id1][obj_id2] = [None, None, None, None]
		
		cls_cluster = self.get_class(self._cm_classes["cluster"])
		cls_node = self.get_class(self._cm_classes["node"])
		cm_names = set([
			primary_class, self._cm_classes["cluster"], self._cm_classes["node"]
		])
		
		if cls_cluster is not None:
			for obj in cls_cluster.get_members(direct_only = True):
				pos = _get_position(obj, descr_pos)
				name = obj.get_descriptor("Name")
				if (pos is None) or (name is None):
					continue
				children = set()
				for obj2, label in obj.get_relations():
					if label != "contains":
						continue
					classes = obj2.get_class_names()
					if (primary_class in classes) and (obj2 not in objects):
						continue
					if cm_names.intersection(classes):
						children.add(obj2.id)
						edges.add((obj.id, obj2.id))
				cluster_data[obj.id] = dict(
					name = name,
					position = pos,
					children = children,
				)
		
		if cls_node is not None:
			for obj in cls_node.get_members(direct_only = True):
				if obj.id in cluster_data:
					continue
				pos = _get_position(obj, descr_pos)
				name = obj.get_descriptor("Name")
				if (pos is None) or (name is None):
					continue
				for obj2, label in obj.get_relations():
					if label != "linked":
						continue
					classes = obj2.get_class_names()
					if (primary_class in classes) and (obj2 not in objects):
						continue
					if cm_names.intersection(classes):
						edges.add((obj.id, obj2.id))
				node_data[obj.id] = dict(
					name = name,
					position = pos,
				)
		
		self._progress.stop()
		QtWidgets.QApplication.restoreOverrideCursor()
		
		self.cmain.cgraph.populate(drawing_data, cluster_data, node_data, edges)
	
	def get_n_samples(self):
		
		return len(self._distance)
	
	def has_distance(self):
		
		for obj_id1 in self._distance:
			for obj_id2 in self._distance[obj_id1]:
				for val in self._distance[obj_id1][obj_id2]:
					if val is not None:
						return True
		return False
	
	def save_distance(self):
		
		self._model.blockSignals(True)
		changed_ids = set()
		for obj_id1 in self._distance:
			obj1 = self.get_object(obj_id1)
			if obj1 is None:
				continue
			changed_ids.add(obj_id1)
			changed_ids.update(self._distance[obj_id1].keys())
			for obj_id2 in self._distance[obj_id1]:
				dists = self._distance[obj_id1][obj_id2]
				for idx, label in enumerate(self._dist_rels):
					if dists[idx] is not None:
						obj1.add_relation(obj_id2, label, weight = dists[idx])
		self._model.blockSignals(False)
		if changed_ids:
			self._model.on_changed(changed_ids)
	
	def delete_distance(self):
		
		self._distance.clear()
		
		objects = self.get_drawing_objects()
		if not objects:
			return
		self._model.blockSignals(True)
		changed_ids = set()
		for obj in objects:
			to_del = set()
			for obj_tgt, label in obj.get_relations():
				if label not in self._dist_rels:
					continue
				to_del.add((obj_tgt.id, label))
				changed_ids.add(obj.id)
				changed_ids.add(obj_tgt.id)
			for obj_tgt_id, label in to_del:
				obj.del_relation(obj_tgt_id, label)
		self._model.blockSignals(False)
		self._model.on_changed(changed_ids)
	
	def add_cluster(self, label, children):
		
		cls_cluster = self.add_class(self._cm_classes["cluster"])
		obj_cluster = cls_cluster.add_member()
		obj_cluster.set_descriptor("Name", label)
		for obj_id in children:
			obj = self.get_object(obj_id)
			if obj is None:
				continue
			obj_cluster.add_relation(obj_id, "contains")
		
		return obj_cluster.id
	
	def add_node(self, label):
		
		cls_node = self.add_class(self._cm_classes["node"])
		obj_node = cls_node.add_member()
		obj_node.set_descriptor("Name", label)
		
		return obj_node.id
	
	def add_link(self, source_id, target_id):
		
		obj_src = self.get_object(source_id)
		obj_tgt = self.get_object(target_id)
		if (obj_src is None) or (obj_tgt is None):
			return
		obj_src.add_relation(target_id, "linked")
	
	def del_link(self, source_id, target_id):
		
		obj_src = self.get_object(source_id)
		obj_tgt = self.get_object(target_id)
		if (obj_src is None) or (obj_tgt is None):
			return
		to_del = set()
		for obj_tgt_, label in obj_src.get_relations():
			if label not in ["linked", "contains"]:
				continue
			if obj_tgt_ != obj_tgt:
				continue
			to_del.add(label)
		for label in to_del:
			obj_src.del_relation(obj_tgt.id, label)
	
	def set_cluster(self, sample_id, cluster_id):
		
		obj_sample = self.get_object(sample_id)
		obj_cluster = self.get_object(cluster_id)
		if (obj_sample is None) or (obj_cluster is None):
			return
		
		self._model.blockSignals(True)
		changed_ids = set([obj_sample.id, obj_cluster.id])
		to_del = set()
		for obj_tgt, label in obj_sample.get_relations():
			if self._cm_classes["cluster"] not in obj_tgt.get_class_names():
				continue
			to_del.add((obj_tgt.id, label))
		for obj_tgt_id, label in to_del:
			obj_sample.del_relation(obj_tgt_id, label)
			changed_ids.add(obj_tgt_id)
		
		obj_cluster.add_relation(sample_id, "contains")
		
		self._model.blockSignals(False)
		self._model.on_changed(changed_ids)
	
	def set_cluster_label(self, obj_id, label):
		
		obj = self.get_object(obj_id)
		if obj is None:
			return
		obj.set_descriptor("Name", label)
	
	def delete_clusters(self, object_ids = None):
		
		all_objects = self.get_drawing_objects()
		
		if object_ids is None:
			objects = all_objects
		else:
			objects = []
			for obj_id in object_ids:
				obj = self.get_object(obj_id)
				if obj is not None:
					objects.append(obj)
		if not objects:
			return
		
		self._model.blockSignals(True)
		deleted_ids = set()
		changed_ids = set()
		
		for obj in objects:
			to_del = set()
			for obj_tgt, label in obj.get_relations():
				if self._cm_classes["cluster"] not in obj_tgt.get_class_names():
					continue
				to_del.add((obj_tgt.id, label))
			for obj_tgt_id, label in to_del:
				obj.del_relation(obj_tgt_id, label)
				changed_ids.add(obj.id)
				changed_ids.add(obj_tgt_id)
		
		cls_cluster = self.get_class(self._cm_classes["cluster"])
		cls_node = self.get_class(self._cm_classes["node"])
		
		if cls_cluster is not None:
			primary_class = self.get_primary_class()
			to_del = set()
			for obj in cls_cluster.get_members(direct_only = True):
				found_rel = False
				for obj_tgt, label in obj.get_relations():
					if primary_class not in obj_tgt.get_class_names():
						continue
					found_rel = True
					break
				if not found_rel:
					to_del.add(obj.id)
			if to_del:
				self.del_objects(to_del)
				deleted_ids.update(to_del)
		
		if cls_node is not None:
			while True:
				to_del = set()
				for obj in cls_node.get_members(direct_only = True):
					found_rel = False
					for obj_tgt, label in obj.get_relations():
						if label in ["linked", "contains"]:
							found_rel = True
							break
					if not found_rel:
						to_del.add(obj.id)
				if to_del:
					self.del_objects(to_del)
					deleted_ids.update(to_del)
				else:
					break
		
		changed_ids = changed_ids.difference(deleted_ids)
		
		self._model.blockSignals(False)
		if deleted_ids:
			self._model.on_deleted(deleted_ids)
		if changed_ids:
			self._model.on_changed(changed_ids)
		
		labels = {}  # {node_id: label, ...}
		objects = []
		_, sample_descr = self.get_cls_descr(self.NAME_ID)
		node_idxs = list(range(len(all_objects)))
		for node_idx, obj in enumerate(all_objects):
			objects.append(obj.id)
			labels[node_idx] = obj.get_descriptor(sample_descr)
		
		all_objects = [obj.id for obj in all_objects]
		self.cmain.cgraph.set_clusters(all_objects, {}, node_idxs, [], labels)
	
	def save_position(self, obj_id, x, y):
		
		descr_pos = self._cm_classes["position"]
		obj = self.get_object(obj_id)
		if obj is None:
			return
		obj.set_geometry_descriptor(self._cm_classes["position"], "POINT", [x, y])
	
	def store_attributes(self, obj_id):
		
		descr_lookup = dict(self.get_descriptors())
		attribute_data = self.cmain.ccontrols.get_attribute_data()
		data = {}
		for name in attribute_data:
			if name in descr_lookup:
				data[(name, descr_lookup[name])] = attribute_data[name]
		
		data[("Date_Modified", descr_lookup["Date_Modified"])] = DDateTime(
			datetime.datetime.now()
		)
		
		# data = {(name, chain): value, key: [{(name, chain): value, ...}, ...], ...}
		self.store_data(data, obj_id, keep_default = True)
		
		self.cmain.cgraph.update_sample_data(obj_id, attribute_data)
	
	
	# ---- Deposit
	# ------------------------------------------------------------------------
	def update_model_info(self):
		
		self.cmain.cview.set_title(self.get_datasource_name())
		self.cmain.ccontrols.set_db_name("%s (%s)" % (
			self.get_datasource_name(),
			str(self.get_datasource()),
		))
		
		path = self.get_folder()
		url = as_url(path)
		if isinstance(self.get_datasource(), Memory):
			path = "temporary"
		self.cmain.ccontrols.set_folder(path, url)
	
	def update_recent(self, kwargs):
		
		datasource = kwargs.get("datasource", None)
		if isinstance(datasource, AbstractDatasource):
			kwargs.update(datasource.to_dict())
		
		url = kwargs.get("url", None)
		if not url:
			path = kwargs.get("path", None)
			if path:
				url = as_url(path)
		self.cmain.cview._view.add_recent_connection(
			url = url,
			identifier = kwargs.get("identifier", None),
			connstr = kwargs.get("connstr", None),
		)
	
	def load(self, *args, **kwargs):
		# datasource = Datasource or format
		
		if not self.cmain.check_save():
			return False
		
		self.clear()
		
		return LCModel.load(self, *args, **kwargs)

