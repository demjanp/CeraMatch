from deposit_gui import AbstractSubcontroller
from deposit_gui.dgui.dgraph_view import DGraphView

from ceramatch.controller.cgraph_nodes.node import Node
from ceramatch.controller.cgraph_nodes.sample_node import SampleNode
from ceramatch.controller.cgraph_nodes.cluster_node import ClusterNode
from ceramatch.controller.cgraph_nodes.tool_tip import ToolTip
from ceramatch.utils.fnc_external import (
	save_xlsx, save_csv,
	import_clusters_xlsx, import_clusters_csv,
)
from ceramatch.utils.fnc_matching import (update_clusters)
from ceramatch.utils.fnc_drawing import (save_catalog)

from deposit.utils.fnc_files import (as_url)

from PySide2 import (QtCore, QtGui, QtWidgets)
import networkx as nx
import numpy as np
import copy
import os

class CGraph(AbstractSubcontroller):
	
	SCALE_DRAWINGS = 0.5
	SCALE_TOOLTIP = 1
	SCALE_CUTOFF = 0.2
	LINE_WIDTH = 1
	
	def __init__(self, cmain):
		
		AbstractSubcontroller.__init__(self, cmain)
		
		self._view = DGraphView()
		self._tool_tip = ToolTip()
		self._descendants = None
		
		self._view.signal_node_activated.connect(self.on_activated)
		self._view.signal_selected.connect(self.on_selected)
	
	
	# ---- Signal handling
	# ------------------------------------------------------------------------
	@QtCore.Slot(object)
	def on_activated(self, node_id):
		
		node = self._view.get_node(node_id)
		if isinstance(node, SampleNode):
			self.cmain.open_deposit("SELECT [%(cls)s].* WHERE [%(cls)s] == %(obj_id)d" % dict(
				cls = self.cmain.cmodel.get_primary_class(),
				obj_id = node_id,
			))
		
		elif isinstance(node, ClusterNode):
			self.cmain.open_deposit("SELECT [%(cls)s].* WHERE [%(cls)s] == %(obj_id)d" % dict(
				cls = self.cmain.cmodel.get_cm_classes()["cluster"],
				obj_id = node_id,
			))
		
		else:
			self.select_descendants()
	
	@QtCore.Slot()
	def on_selected(self):
		
		self.cmain.ccontrols.update()
		
		self.cmain.ccontrols.set_attribute_data({})
		
		node_ids, _ = self._view.get_selected()
		if len(node_ids) == 1:
			node = self._view.get_node(node_ids[0])
			if isinstance(node, SampleNode):
				self.cmain.ccontrols.set_attribute_data(node.get_drawing_data())
		
		self.cmain.cactions.update()
	
	def on_hover(self, node, state):
		
		tool_tip = node.get_tool_tip()
		if tool_tip is None:
			self._tool_tip.hide()
		else:
			self._tool_tip.show(tool_tip, node)
	
	def on_moved(self, node):
		
		pos = node.pos()
		self.cmain.cmodel.save_position(node.node_id, pos.x(), pos.y())
		self.cmain.history.save()
	
	def on_mouse_released(self, node):
		
		pass
	
	def on_drop(self, src_node, tgt_node):
		
		if isinstance(src_node, SampleNode) and isinstance(tgt_node, ClusterNode):
			node_ids, _ = self._view.get_selected()
			src_nodes = []
			for node_id in node_ids:
				node = self._view.get_node(node_id)
				if not isinstance(node, SampleNode):
					continue
				src_nodes.append(node)
			if not src_nodes:
				return
			
			self.add_to_cluster(src_nodes, tgt_node)
			self.cmain.history.save()
		
		if isinstance(src_node, ClusterNode):
			if isinstance(tgt_node, ClusterNode):
				self.merge_clusters(src_node, tgt_node)
				self.cmain.history.save()
			elif tgt_node.__class__ == Node:
				self.reparent_cluster(src_node, tgt_node)
				self.cmain.history.save()
	
	
	# ---- get/set
	# ------------------------------------------------------------------------
	def get_selected_samples(self):
		
		node_ids, _ = self._view.get_selected()
		selected = []
		for node_id in node_ids:
			node = self._view.get_node(node_id)
			if not isinstance(node, SampleNode):
				continue
			selected.append(node_id)
		
		return selected
	
	def get_selected_clusters(self):
		
		node_ids, _ = self._view.get_selected()
		selected = []
		for node_id in node_ids:
			node = self._view.get_node(node_id)
			if not isinstance(node, ClusterNode):
				continue
			selected.append(node_id)
		
		return selected
	
	def get_selected(self):
		
		node_ids, _ = self._view.get_selected()
		
		return node_ids
	
	def get_scale_factor(self):
		
		return self._view.get_scale_factor()
	
	def clear(self):
		
		self._descendants = None
		self._view.clear()
	
	def populate(self, drawing_data, cluster_data, node_data, edges):
		# drawing_data = {obj_id: {key: value, key: [{key: value, ...}, ...], ...}, ...}
		# cluster_data = {obj_id: {"name": value, "position": (x, y), "children": [obj_id, ...], ...}, ...}
		# node_data = {obj_id: {"name": value, "position": (x, y), ...}, ...}
		# edges = [(source_id, target_id), ...]
		
		nodes = []  # [SampleNode, ClusterNode, ...]
		positions = {}  # {node_id: (x, y), ...}
		move_nodes = []  # [node_id, ...]; if set, only calculate positions for those nodes
		
		self._descendants = None
		
		# prune clusters & nodes
		collect = {}
		for obj_id in cluster_data:
			if cluster_data[obj_id]["children"]:
				collect[obj_id] = cluster_data[obj_id]
		cluster_data = collect
		G = nx.DiGraph()
		for source_id, target_id in edges:
			G.add_edge(source_id, target_id)
		collect = {}
		cluster_ids = set(cluster_data.keys())
		for obj_id in node_data:
			if cluster_ids.intersection(nx.descendants(G, obj_id)):
				collect[obj_id] = node_data[obj_id]
		node_data = collect
		
		self.cmain.cview.progress.show("Rendering Drawings")
		self.cmain.cview.progress.update_state(value = 0, maximum = len(drawing_data))
		cnt = 1
		for obj_id in drawing_data:
			self.cmain.cview.progress.update_state(value = cnt)
			if self.cmain.cview.progress.cancel_pressed():
				return
			cnt += 1
			nodes.append(SampleNode(self,
				node_id = obj_id,
				label = drawing_data[obj_id][self.cmain.cmodel.NAME_ID],
				drawing_data = drawing_data[obj_id],
			))
			pos = drawing_data[obj_id].get("position")
			if pos is not None:
				positions[obj_id] = pos
		
		for obj_id in cluster_data:
			nodes.append(ClusterNode(self,
				node_id = obj_id,
				label = cluster_data[obj_id]["name"],
				children = cluster_data[obj_id]["children"],
			))
			positions[obj_id] = cluster_data[obj_id]["position"]
		
		for obj_id in node_data:
			nodes.append(Node(self,
				node_id = obj_id,
				label = node_data[obj_id]["name"],
			))
			positions[obj_id] = node_data[obj_id]["position"]
		
		self.cmain.cview.set_dummy_graph(True)
		
		self._view.populate(
			nodes, edges, positions,
			x_stretch = 1,
			y_stretch = 2,
			edge_arrow_size = 0,
			move_nodes = move_nodes,
			progress = self.cmain.cview.progress,
		)
		
		for node in self._view.get_nodes():
			node._moved = False
		
		self.cmain.cview.set_dummy_graph(False)
	
	def set_clusters(self, objects, clusters, node_idxs, edges, labels, positions = {}):
		# objects = [obj_id, ...]
		# clusters = {node_idx: [object_idx, ...], ...}; object_idx = index in objects
		# node_idxs = [node_idx, ...]
		# edges = [(source_idx, target_idx), ...]
		# labels = {node_idx: label, ...}
		# positions = {node_idx: (x, y), ...}
		
		self.cmain.cmodel._model.blockSignals(True)
		
		added_ids = set()
		changed_ids = set()
		node_id_lookup = {}
		self._descendants = None
		
		nodes = []
		for node in self._view.get_nodes():
			if not isinstance(node, SampleNode):
				continue
			if node.node_id not in objects:
				continue
			nodes.append(node.copy())
			node_id_lookup[objects.index(node.node_id)] = node.node_id
		
		for node_idx in list(clusters.keys()):
			children = [objects[idx] for idx in clusters[node_idx] if idx in node_id_lookup]
			if not children:
				del clusters[node_idx]
				continue
			node_id = self.cmain.cmodel.add_cluster(labels[node_idx], children)
			added_ids.add(node_id)
			node_id_lookup[node_idx] = node_id
			nodes.append(ClusterNode(self,
				node_id = node_id,
				label = labels[node_idx],
				children = children,
			))
		
		G = nx.DiGraph()
		for source_idx, target_idx in edges:
			G.add_edge(source_idx, target_idx)
		cluster_idxs = set(clusters.keys())
		
		for node_idx in node_idxs:
			if node_idx in node_id_lookup:
				continue
			if node_idx < len(objects):
				continue
			if not cluster_idxs.intersection(nx.descendants(G, node_idx)):
				continue
			node_id = self.cmain.cmodel.add_node(labels[node_idx])
			added_ids.add(node_id)
			node_id_lookup[node_idx] = node_id
			nodes.append(Node(self,
				node_id = node_id,
				label = labels[node_idx],
			))
		
		edges_ = []
		for source_idx, target_idx in edges:
			if (source_idx not in node_id_lookup) or (target_idx not in node_id_lookup):
				continue
			source_id = node_id_lookup[source_idx]
			target_id = node_id_lookup[target_idx]
			edges_.append([source_id, target_id])
			self.cmain.cmodel.add_link(source_id, target_id)
			changed_ids.add(source_id)
			changed_ids.add(target_id)
		
		self.cmain.cview.set_dummy_graph(True)
		
		self._view.populate(
			nodes = nodes, 
			edges = edges_, 
			positions = dict([(node_id_lookup[node_idx], positions[node_idx]) \
				for node_idx in positions]),
			x_stretch = 1,
			y_stretch = 1 if positions else 2,
			edge_arrow_size = 0,
			progress = self.cmain.cview.progress,
		)
		
		self.cmain.cview.set_dummy_graph(False)
		
		for node in self._view.get_nodes():
			pos = node.pos()
			self.cmain.cmodel.save_position(node.node_id, pos.x(), pos.y())
			node._moved = False
		
		self.cmain.cmodel._model.blockSignals(False)
		self.cmain.cmodel._model.on_added(added_ids)
		self.cmain.cmodel._model.on_changed(changed_ids)
	
	def get_cluster_data(self):
		# returns objects, clusters, node_idxs, edges, labels, positions
		#	objects = [obj_id, ...]
		#	clusters = {node_idx: [object_idx, ...], ...}; object_idx = index in objects
		#	node_idxs = [node_idx, ...]
		#	edges = [(source_idx, target_idx), ...]
		#	labels = {node_idx: label, ...}
		#	positions = {node_idx: (x, y), ...}
		
		objects = [obj.id for obj in self.cmain.cmodel.get_drawing_objects()]
		nodes = objects.copy()
		for node in self._view.get_nodes():
			if node.node_id not in nodes:
				nodes.append(node.node_id)
		node_idxs = list(range(len(nodes)))
		clusters = self.get_clusters()
		edges = []
		for edge in self._view.get_edges():
			edges.append((
				nodes.index(edge.source().node_id), 
				nodes.index(edge.target().node_id),
			))
		labels = {}
		positions = {}
		for node in self._view.get_nodes():
			node_idx = nodes.index(node.node_id)
			labels[node_idx] = node.label
			pos = node.pos()
			positions[node_idx] = (pos.x(), pos.y())
		clusters = dict([(
			nodes.index(node_id),
			[nodes.index(sample_id) for sample_id in clusters[node_id]],
		) for node_id in clusters])
		
		return objects, clusters, node_idxs, edges, labels, positions
	
	def get_descendants(self, node_id):
		
		if self._descendants is None:
			G = nx.DiGraph()
			for edge in self._view.get_edges():
				G.add_edge(edge.source().node_id, edge.target().node_id)
			self._descendants = {}
			for node_id_ in G.nodes():
				self._descendants[node_id_] = set(nx.descendants(G, node_id_))
		
		return self._descendants[node_id]
	
	def select_descendants(self):
		
		node_ids = set()
		for node_id in self.get_selected():
			node_ids.update(self.get_descendants(node_id))
		cnt = 1
		self._view.blockSignals(True)
		for node_id in node_ids:
			cnt += 1
			self._view.select_node(node_id)
		self._view.blockSignals(False)
	
	def del_node(self, node_id):
		
		self.cmain.cmodel.del_object(node_id)
		self._view.del_node(node_id)
	
	def update_sample_data(self, node_id, data):
		# data = {name: value, ...}
		
		node = self._view.get_node(node_id)
		if not isinstance(node, SampleNode):
			return
		drawing_data = node.get_drawing_data()
		drawing_data.update(data)
		node.set_drawing_data(data)
	
	def add_to_cluster(self, src_nodes, tgt_node):
		
		self._view.del_edges([node.node_id for node in src_nodes])
		
		for src_node in src_nodes:
			self._view.add_edge(
				tgt_node.node_id, src_node.node_id,
				color = QtCore.Qt.black,
				arrow_size = 0,
			)
			self.cmain.cmodel.set_cluster(src_node.node_id, tgt_node.node_id)
		
		self.prune_tree()
	
	def merge_clusters(self, src_node, tgt_node):
		
		src_node = self._view.get_node(src_node.node_id)
		tgt_node = self._view.get_node(tgt_node.node_id)
		src_nodes = src_node.get_children()
		self.del_node(src_node.node_id)
		self.add_to_cluster(src_nodes, tgt_node)
	
	def reparent_cluster(self, src_node, tgt_node):
		
		self.cmain.cmodel._model.blockSignals(True)
		changed_ids = set()
		
		del_links = set()
		for parent in src_node.get_parents():
			del_links.add((parent.node_id, src_node.node_id))
		for src_id, tgt_id in del_links:
			self._view.del_edge(src_id, tgt_id)
			self.cmain.cmodel.del_link(src_id, tgt_id)
			changed_ids.add(src_id)
			changed_ids.add(tgt_id)
		self._view.add_edge(
			tgt_node.node_id,
			src_node.node_id,
			color = QtCore.Qt.black,
			arrow_size = 0,
		)
		self.cmain.cmodel.add_link(src_node.node_id, tgt_node.node_id)
		changed_ids.add(src_node.node_id)
		changed_ids.add(tgt_node.node_id)
		
		self.cmain.cmodel._model.blockSignals(False)
		self.cmain.cmodel._model.on_changed(changed_ids)
		
		self.prune_tree()
	
	def add_manual_cluster(self):
		
		sample_ids = self.get_selected_samples()
		if not sample_ids:
			return
		
		self.cmain.cmodel._model.blockSignals(True)
		changed_ids = set(sample_ids)
		
		x = 0
		y = np.inf
		parent_ids = set()
		for node_id in sample_ids:
			node = self._view.get_node(node_id)
			pos = node.pos()
			x += pos.x()
			y = min(y, pos.y())
			to_del = []
			for node_ in node.get_parents():
				for parent in node_.get_parents():
					parent_ids.add(parent.node_id)
				self._view.del_edge(node_.node_id, node_id)
		parent_id = None
		if parent_ids:
			parent_id = min(parent_ids)
		x /= len(sample_ids)
		y -= 10
		label = "Manual_%d" % (node_id)
		cluster_id = self.cmain.cmodel.add_cluster(label, sample_ids)
		cluster = ClusterNode(self,
			node_id = cluster_id,
			label = label,
			children = sample_ids,
		)
		self._view.add_node(cluster, (x, y))
		for sample_id in sample_ids:
			self._view.add_edge(cluster_id, sample_id, color = QtCore.Qt.black, arrow_size = 0)
		if parent_id is not None:
			self._view.add_edge(parent_id, cluster_id, color = QtCore.Qt.black, arrow_size = 0)
			self.cmain.cmodel.add_link(parent_id, cluster_id)
		
		self.cmain.cmodel._model.blockSignals(False)
		self.cmain.cmodel._model.on_changed(changed_ids)
		self.cmain.cmodel._model.on_added([cluster_id])
		
		self.prune_tree()
	
	def rename_cluster(self):
		
		obj_id = self.get_selected_clusters()
		if len(obj_id) != 1:
			return
		obj_id = obj_id[0]
		node = self._view.get_node(obj_id)
		label = self.cmain.cview.show_input_dialog("Rename Cluster", "Label:", node.label)
		node.set_label(label)
		self.cmain.cmodel.set_cluster_label(obj_id, label)
	
	def prune_tree(self):
		
		self.cmain.cmodel._model.blockSignals(True)
		deleted_ids = set()
		
		to_del = []
		for node in self._view.get_nodes():
			if not (isinstance(node, ClusterNode) or node.__class__ == Node):
				continue
			if not node.has_child():
				to_del.append(node)
		for node in to_del:
			deleted_ids.add(node.node_id)
			self.del_node(node.node_id)
		
		self.cmain.cmodel._model.blockSignals(False)
		self.cmain.cmodel._model.on_deleted(deleted_ids)
	
	def export_clusters(self, path, format):
		
		header = ["Sample ID", "Cluster"]
		rows = []
		for node in self._view.get_nodes():
			if not isinstance(node, ClusterNode):
				continue
			cluster = node.label
			for sample in node.get_children():
				rows.append([sample.label, cluster])
		
		if format == "Excel 2007+ Workbook (*.xlsx)":
			save_xlsx(header, rows, path)
		
		elif format == "Comma-separated Values (*.csv)":
			save_csv(header, rows, path)
		
		self.cmain.cview.show_notification(
			'''
Exported to: <a href="%s">%s</a>
			''' % (as_url(path), path),
			delay = 7000,
		)
	
	def import_clusters(self, path, sample_column, cluster_column):
		
		format = os.path.splitext(path)[-1].strip(".").lower()
		clusters = {}
		if format == "xlsx":
			clusters = import_clusters_xlsx(path, sample_column, cluster_column)
		elif format == "csv":
			clusters = import_clusters_csv(path, sample_column, cluster_column)
		if not clusters:
			return
		
		sample_id_lookup = {}
		for node in self._view.get_nodes():
			if not isinstance(node, SampleNode):
				continue
			sample_id_lookup[node.label] = node.node_id
		collect = {}
		labels = {}
		idx = 0
		for cluster_label in clusters:
			samples = []
			for label in clusters[cluster_label]:
				if label not in sample_id_lookup:
					continue
				samples.append(sample_id_lookup[label])
			if samples:
				collect[idx] = samples.copy()
				labels[idx] = cluster_label
				idx += 1
		clusters = collect
		if not clusters:
			return		
		
		self.cmain.cview.progress.show("Clustering")
		objects, clusters, nodes, edges, labels = update_clusters(
			self.cmain.cmodel._distance,
			clusters = clusters,
			labels = labels,
			progress = self.cmain.cview.progress,
		)
		self.cmain.cview.progress.stop()
		
		self.cmain.cmodel.delete_clusters()
		self.set_clusters(objects, clusters, nodes, edges, labels)
		self.cmain.history.save()
		self.cmain.ccontrols.update()
	
	def export_dendrogram(self, path, dpi, page_size, stroke_width):
		
		scale = self._view.get_scale_factor()
		self._view.scale_view(1/scale)
		self._view.save_pdf(path, dpi, page_size, stroke_width)
		self._view.scale_view(scale)
		self.cmain.cview.show_notification(
			'''
Dendrogram exported to: <a href="%s">%s</a>
			''' % (as_url(path), path),
			delay = 7000,
		)
	
	def export_catalog(self, path, scale, page_size, stroke_width):
		
		self.cmain.cview.progress.show("Saving Catalog")
		
		data = {}
		clusters = {}
		labels = {}
		for node in self._view.get_nodes():
			labels[node.node_id] = node.label
			if isinstance(node, SampleNode):
				data[node.node_id] = node.get_drawing_data()
			elif isinstance(node, ClusterNode):
				clusters[node.node_id] = [
					child.node_id for child in node.get_children()
				]
		# data = {sample_id: {key: value, key: [{key: value, ...}, ...], ...}, ...}
		# clusters = {obj_id: [sample_id, ...], ...}
		# labels = {obj_id: label, ...}
		
		save_catalog(
			path, data, clusters, labels, scale, page_size, stroke_width, 
			progress = self.cmain.cview.progress,
		)
		self.cmain.cview.progress.stop()
		
		self.cmain.cview.show_notification(
			'''
Catalog exported to: <a href="%s">%s</a>
			''' % (as_url(path), path),
			delay = 7000,
		)
	
	def has_drawings(self):
		
		return self._view.has_nodes()
	
	def has_clusters(self):
		
		for node in self._view.get_nodes():
			if not isinstance(node, SampleNode):
				return True
		return False
	
	def get_n_clusters(self):
		
		n_clusters = 0
		for node in self._view.get_nodes():
			if isinstance(node, ClusterNode):
				n_clusters += 1
		
		return n_clusters
	
	def get_clusters(self):
		# returns clusters = {node_idx: [obj_id, ...], ...}
		
		clusters = {}
		for node in self._view.get_nodes():
			if not isinstance(node, ClusterNode):
				continue
			clusters[node.node_id] = [
				child.node_id for child in node.get_children()
			]
		return clusters
	
	def get_cluster_labels(self):
		
		labels = {}
		for node in self._view.get_nodes():
			if not isinstance(node, ClusterNode):
				continue
			labels[node.node_id] = node.label
		return labels
	
	def get_profile_data(self):
		# returns {obj_id: (coords, radius), ...}
		
		data = {}
		for node in self._view.get_nodes():
			if not isinstance(node, SampleNode):
				continue
			coords, radius = node.get_profile_data()
			if radius is None:
				continue
			data[node.node_id] = (coords, radius)
		
		return data

