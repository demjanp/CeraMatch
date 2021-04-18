from lib.Model import (Model)
from lib.dialogs._Dialogs import (Dialogs)
from lib.ToolBar import (ToolBar)
from lib.Menu import (Menu)
from lib.DescriptorGroup import (DescriptorGroup)
from lib.DistanceGroup import (DistanceGroup)
from lib.ClusterGroup import (ClusterGroup)
from lib.GraphView import (GraphView)
from lib.StatusBar import (StatusBar)

from deposit import Broadcasts
from deposit.commander.Registry import (Registry)
from deposit.DModule import (DModule)

from PySide2 import (QtWidgets, QtCore, QtGui)
import os

class View(DModule, QtWidgets.QMainWindow):
	
	def __init__(self):
		
		self.model = None
		
		DModule.__init__(self)
		QtWidgets.QMainWindow.__init__(self)
		
		self.model = Model(self)
		
		self.dialogs = Dialogs(self)
		self.registry = Registry("Deposit")
		self.graph_view = GraphView(self)
		self.descriptor_group = DescriptorGroup(self)
		self.distance_group = DistanceGroup(self)
		self.cluster_group = ClusterGroup(self)
		self.menu = Menu(self)
		self.toolbar = ToolBar(self)
		self.statusbar = StatusBar(self)
		self.progress = None
		
		self.setWindowIcon(QtGui.QIcon("res\cm_icon.svg"))
		self.setStyleSheet("QPushButton {padding: 5px; min-width: 100px;}")
		
		central_widget = QtWidgets.QWidget(self)
		central_widget.setLayout(QtWidgets.QHBoxLayout())
		central_widget.layout().setContentsMargins(0, 0, 0, 0)
		self.setCentralWidget(central_widget)
		
		control_frame = QtWidgets.QFrame(self)
		control_frame.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
		control_frame.setLayout(QtWidgets.QVBoxLayout())
		control_frame.layout().setContentsMargins(10, 10, 0, 10)
		
		graph_view_frame = QtWidgets.QFrame(self)
		graph_view_frame.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
		graph_view_frame.setLayout(QtWidgets.QVBoxLayout())
		graph_view_frame.layout().setContentsMargins(0, 0, 0, 0)
		
		central_widget.layout().addWidget(control_frame)
		central_widget.layout().addWidget(graph_view_frame)
		
		control_frame.layout().addWidget(self.descriptor_group)
		control_frame.layout().addWidget(self.distance_group)
		control_frame.layout().addWidget(self.cluster_group)
		control_frame.layout().addStretch()
		
		graph_view_frame.layout().addWidget(self.graph_view)
		
		self.setStatusBar(self.statusbar)
		
		self.set_title()
#		self.setGeometry(100, 100, 1024, 768)
		self.setGeometry(500, 100, 1024, 768)  # DEBUG
		
		self.descriptor_group.load_data.connect(self.on_load_data)
		self.descriptor_group.cluster_classes_changed.connect(self.on_cluster_classes_changed)
		self.distance_group.calculate.connect(self.on_calculate)
		self.distance_group.delete.connect(self.on_delete_distance)
		self.cluster_group.cluster.connect(self.on_cluster)
		self.cluster_group.update_tree.connect(self.on_update_tree)
		self.cluster_group.add_cluster.connect(self.on_add_cluster)
		self.cluster_group.delete.connect(self.on_delete_clusters)
		
		self.connect_broadcast(Broadcasts.VIEW_ACTION, self.on_view_action)
		self.connect_broadcast(Broadcasts.STORE_LOADED, self.on_data_source_changed)
		self.connect_broadcast(Broadcasts.STORE_DATA_SOURCE_CHANGED, self.on_data_source_changed)
		self.connect_broadcast(Broadcasts.STORE_DATA_CHANGED, self.on_data_changed)
		self.connect_broadcast(Broadcasts.STORE_SAVED, self.on_saved)
		self.connect_broadcast(Broadcasts.STORE_SAVE_FAILED, self.on_save_failed)
		self.set_on_broadcast(self.on_broadcast)
		
		self.model.broadcast_timer.setSingleShot(True)
		self.model.broadcast_timer.timeout.connect(self.on_broadcast_timer)
		
		self.update()
		
		self.dialogs.open("Connect")
	
	def set_title(self, name = None):

		title = "CeraMatch"
		if name is None:
			self.setWindowTitle(title)
		else:
			self.setWindowTitle("%s - %s" % (name, title))
	
	def show_progress(self, text):
		
		self.progress = QtWidgets.QProgressDialog(text, None, 0, 0, self, flags = QtCore.Qt.FramelessWindowHint)
		self.progress.setWindowModality(QtCore.Qt.WindowModal)
		self.progress.show()
		QtWidgets.QApplication.processEvents()
	
	def hide_progress(self):
		
		self.progress.hide()
		self.progress.setParent(None)
	
	def update(self):
		
		self.set_title(os.path.split(str(self.model.identifier))[-1].strip("#"))
		self.descriptor_group.update()
		self.distance_group.update()
		self.cluster_group.update()
	
	def save(self):
		
		if self.model.data_source is None:
			self.dialogs.open("Connect")
			
		else:
			self.show_progress("Saving...")
			self.model.save()
			self.hide_progress()
	
	@QtCore.Slot()
	def on_load_data(self):
		
		self.show_progress("Loading...")
		nodes = None
		if self.model.lap_descriptors is not None:
			clusters, nodes, edges, labels = self.model.load_samples()
		
		if nodes is None:
			self.graph_view.set_data(self.model.sample_data)
			self.cluster_group.update_clusters_found(None)
		else:
			self.cluster_group.update_clusters_found(len(clusters))
			self.graph_view.set_data(self.model.sample_data, clusters, nodes, edges, labels)
		
		self.descriptor_group.update()
		self.distance_group.update()
		self.cluster_group.update()
		self.cluster_group.update_n_clusters()
		
		self.update()
		self.hide_progress()
	
	@QtCore.Slot()
	def on_cluster_classes_changed(self):
		
		self.cluster_group.update()
	
	@QtCore.Slot()
	def on_cluster(self):
		
		self.show_progress("Clustering...")
		self.cluster_group.update_clusters_found(None)
		max_clusters, limit = self.cluster_group.get_limits()
		clusters, nodes, edges, labels = self.model.clusters.make(max_clusters, limit)
		if nodes is not None:
			self.cluster_group.update_clusters_found(len(clusters))
			self.graph_view.set_data(self.model.sample_data, clusters, nodes, edges, labels)
		self.update()
		self.hide_progress()
	
	@QtCore.Slot()
	def on_update_tree(self):
		
		self.show_progress("Updating...")
		clusters, nodes, edges, labels = self.model.clusters.update()
		if nodes is None:
			self.graph_view.set_data(self.model.sample_data)
			self.cluster_group.update_clusters_found(None)			
		else:
			self.graph_view.set_data(self.model.sample_data, clusters, nodes, edges, labels)
			self.cluster_group.update_clusters_found(len(clusters))
		self.update()
		self.hide_progress()
	
	@QtCore.Slot()
	def on_add_cluster(self):
		
		self.graph_view.add_cluster()
	
	@QtCore.Slot()
	def on_calculate(self):
		
		self.show_progress("Calculating...")
		self.model.calc_distance()
		self.update()
		self.hide_progress()
	
	@QtCore.Slot()
	def on_delete_distance(self):
		
		reply = QtWidgets.QMessageBox.question(self, "Delete Distances", "Delete distances from database?", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
		if reply == QtWidgets.QMessageBox.Yes:
			self.show_progress("Deleting...")
			self.model.delete_distance()
			self.update()
			self.hide_progress()
	
	@QtCore.Slot()
	def on_delete_clusters(self):
		
		reply = QtWidgets.QMessageBox.question(self, "Delete Clusters", "Delete clusters from database?", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
		if reply == QtWidgets.QMessageBox.Yes:
			self.show_progress("Deleting...")
			self.model.clusters.delete()
			labels = dict([(str(sample_id), sample_id) for sample_id in self.model.sample_ids])
			self.graph_view.set_data(self.model.sample_data)
			self.update()
			self.hide_progress()
	
	def on_view_action(self, *args):
		
		pass
	
	def on_broadcast(self, signals):
		
		if (Broadcasts.STORE_SAVED in signals) or (Broadcasts.STORE_SAVE_FAILED in signals):
			self.process_broadcasts()
		else:
			self.model.broadcast_timer.start(100)
	
	def on_broadcast_timer(self):

		self.process_broadcasts()
	
	def on_data_source_changed(self, *args):
		
		self.set_title(os.path.split(str(self.model.identifier))[-1].strip("#"))
		self.statusbar.message("")
	
	def on_data_changed(self, *args):
		
		self.statusbar.message("")
	
	def on_saved(self, *args):
		
		self.statusbar.message("Database saved.")
	
	def on_save_failed(self, *args):
		
		self.statusbar.message("Saving failed!")
		
