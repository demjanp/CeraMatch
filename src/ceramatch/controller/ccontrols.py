from deposit_gui import AbstractSubcontroller

from ceramatch.view.vcontrols import VControls

from ceramatch.utils.fnc_matching import (get_clusters, update_clusters)

from PySide2 import (QtCore)

class CControls(AbstractSubcontroller):
	
	def __init__(self, cmain) -> None:
		
		AbstractSubcontroller.__init__(self, cmain)
		
		self._view = VControls()
		
		self._view.signal_folder_link_clicked.connect(self.on_folder_link_clicked)
		
		self._view.descriptors.signal_load_drawings.connect(self.on_load_drawings)
		self._view.descriptors.signal_cluster_classes_changed.connect(
			self.on_cluster_classes_changed
		)
		
		self._view.distances.signal_calculate.connect(self.on_calculate_distances)
		self._view.distances.signal_delete.connect(self.on_delete_distances)
		
		self._view.clusters.signal_cluster.connect(self.on_cluster)
		self._view.clusters.signal_update_tree.connect(self.on_update_tree)
		self._view.clusters.signal_add_cluster.connect(self.on_add_cluster)
		self._view.clusters.signal_rename_cluster.connect(self.on_rename_cluster)
		self._view.clusters.signal_delete.connect(self.on_delete_clusters)
		self._view.clusters.signal_n_clusters_changed.connect(self.on_n_clusters_changed)
		self._view.attributes.signal_store_attributes.connect(self.on_store_attributes)
	
	
	# ---- Signal handling
	# ------------------------------------------------------------------------
	def on_loaded(self):
		
		descriptors = self.cmain.cmodel.get_descriptors()
		attributes = self.cmain.cmodel.get_attributes()
		cls = self.cmain.cmodel.get_primary_class()
		classes = self.cmain.cmodel.get_class_names(ordered = True)
		
		cm_classes = self.cmain.cmodel.get_cm_classes()
		
		descriptors_ = {}  # {label: chain, ...}
		name_lookup = dict([(name, label) for label, _, name in attributes])
		for name, chain in descriptors:
			descriptors_[name_lookup.get(name, name)] = chain
		
		self._view.descriptors.set_descriptors(descriptors_)
		self._view.descriptors.set_cluster_class(cm_classes["cluster"], classes)
		self._view.descriptors.set_node_class(cm_classes["node"], classes)
		self._view.descriptors.set_position_class(cm_classes["position"], classes)
		
		self.update()
	
	@QtCore.Slot(str)
	def on_folder_link_clicked(self, path):
		
		self.cmain.open_folder(path)
	
	@QtCore.Slot()
	def on_load_drawings(self):
		
		self.cmain.history.clear()
		self.cmain.cmodel.load_drawings()
		self.cmain.history.save()
		self.update()
	
	@QtCore.Slot()
	def on_cluster_classes_changed(self):
		
		self.cmain.cmodel.set_cm_classes(dict(
			cluster = self._view.descriptors.get_cluster_class(),
			node = self._view.descriptors.get_node_class(),
			position = self._view.descriptors.get_position_class(),
		))
	
	@QtCore.Slot()
	def on_calculate_distances(self):
		
		self.cmain.cdialogs.open("SelectDistances")
	
	@QtCore.Slot()
	def on_delete_distances(self):
		
		if not self.cmain.cview.show_question(
			"Delete Distances",
			"Are you sure to delete distances from database?\nWarning: This action cannot be undone!",
		):
			return
		self.cmain.cmodel.delete_distance()
		self.update()
	
	@QtCore.Slot()
	def on_cluster(self):
		
		n_clusters, limit = self._view.clusters.get_limits()
		self.cmain.cview.progress.show("Clustering")
		objects, clusters, nodes, edges, labels = get_clusters(
			self.cmain.cmodel._distance,
			max_clusters = n_clusters,
			limit = limit,
			progress = self.cmain.cview.progress,
		)
		self.cmain.cview.progress.stop()
		
		self.cmain.cmodel.delete_clusters()
		self.cmain.cgraph.set_clusters(objects, clusters, nodes, edges, labels)
		self.cmain.history.save()
		
		self.update()
	
	@QtCore.Slot()
	def on_update_tree(self):
		
		self.cmain.cview.progress.show("Clustering")
		objects, clusters, nodes, edges, labels = update_clusters(
			self.cmain.cmodel._distance,
			clusters = self.cmain.cgraph.get_clusters(),
			labels = self.cmain.cgraph.get_cluster_labels(),
			progress = self.cmain.cview.progress,
		)
		self.cmain.cview.progress.stop()
		
		self.cmain.cmodel.delete_clusters()
		self.cmain.cgraph.set_clusters(objects, clusters, nodes, edges, labels)
		self.cmain.history.save()
		
		self.update()
	
	@QtCore.Slot()
	def on_add_cluster(self):
		
		self.cmain.cgraph.add_manual_cluster()
		self.cmain.history.save()
	
	@QtCore.Slot()
	def on_rename_cluster(self):
		
		self.cmain.cgraph.rename_cluster()
		self.cmain.history.save()
	
	@QtCore.Slot()
	def on_delete_clusters(self):
		
		self.cmain.cmodel.delete_clusters()
		self.cmain.history.save()
		self.update()
	
	@QtCore.Slot()
	def on_n_clusters_changed(self):
		
		self.update()
	
	@QtCore.Slot()
	def on_store_attributes(self):
		
		obj_id = self.cmain.cgraph.get_selected_samples()
		if len(obj_id) != 1:
			return
		obj_id = obj_id[0]
		self.cmain.cmodel.store_attributes(obj_id)
		self.update_attribute_data()
	
	
	# ---- get/set
	# ------------------------------------------------------------------------
	def clear(self):
		
		self._view.descriptors.clear()
	
	def update(self):
		
		n_clusters, _ = self._view.clusters.get_limits()
		n_max = self.cmain.cmodel.get_n_samples()
		clustering_enabled = (n_max > 2)
		n_selected = len(self.cmain.cgraph.get_selected_samples())
		
		if clustering_enabled:
			values = [""] + list(range(2, n_max))
			self._view.clusters.set_n_clusters(values, n_clusters)
		else:
			self._view.clusters.set_n_clusters([])
		
		self._view.clusters.set_n_samples(n_max)
		
		self._view.clusters.set_n_found(self.cmain.cgraph.get_n_clusters())
		
		self._view.descriptors.set_load_drawings_enabled(True)
		self._view.distances.set_calculate_distances_enabled(self.cmain.cgraph.has_drawings())
		self._view.distances.set_delete_distances_enabled(self.cmain.cmodel.has_distance())
		self._view.clusters.set_clustering_enabled(clustering_enabled)
		self._view.clusters.set_limit_enabled(clustering_enabled & (n_clusters is None))
		self._view.clusters.set_update_tree_enabled(self.cmain.cgraph.has_clusters())
		self._view.clusters.set_add_cluster_enabled(n_selected > 0)
		self._view.clusters.set_rename_cluster_enabled(len(self.cmain.cgraph.get_selected_clusters()) == 1)
		self._view.clusters.set_delete_enabled(self.cmain.cgraph.has_clusters())
	
	def set_db_name(self, name):
		
		self._view.set_db_name(name)
	
	def set_folder(self, path, url = None):
		
		self._view.set_folder(path, url)
	
	def set_attributes(self, rows):
		# rows = [(label, ctrl_type, name), ...]
		
		self._view.attributes.populate(rows)
	
	def set_attribute_data(self, data):
		# return data = {name: value, ...}
		
		# data = {name: (value, items), ...}; items = [value, ...]
		values = self.cmain.cmodel.get_descriptor_values()
		attr_data = {}
		for name in data:
			attr_data[name] = (data[name], [])
		for name in values:
			attr_data[name] = (attr_data.get(name, (None, []))[0], values[name])
		
		self._view.attributes.set_data(attr_data)
	
	def get_attribute_data(self):
		# return data = {name: value, ...}
		
		return self._view.attributes.get_data()
	
	def update_attribute_data(self):
		
		self.set_attribute_data(self.get_attribute_data())

