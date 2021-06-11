
from lib.fnc_drawing import *

from lib.Model import (Model)
from lib.dialogs._Dialogs import (Dialogs)
from lib.toolbar._Toolbar import (Toolbar)
from lib.menu._Menu import Menu
from lib.DescriptorGroup import (DescriptorGroup)
from lib.DistanceGroup import (DistanceGroup)
from lib.ClusterGroup import (ClusterGroup)
from lib.GraphView import (GraphView)
from lib.StatusBar import (StatusBar)
from lib.Progress import (Progress)

from deposit import Broadcasts
from deposit.commander.Registry import (Registry)
from deposit.DModule import (DModule)

from PySide2 import (QtWidgets, QtCore, QtGui)
import deposit
import res
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
		self.toolbar = Toolbar(self)
		self.menu = Menu(self)
		self.statusbar = StatusBar(self)
		self.progress = Progress(self)
		
		self.setWindowIcon(self.get_icon("cm_icon.svg"))
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
		
		self.menu.load_recent()
		
		self.set_title()
		self.setGeometry(100, 100, 1024, 768)
#		self.setGeometry(500, 100, 1024, 768)  # DEBUG
		
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
	
	def update_mrud(self):
		
		if self.model.data_source is None:
			return
		if self.model.data_source.connstr is None:
			self.menu.add_recent_url(self.model.data_source.url)
		else:
			self.menu.add_recent_db(self.model.data_source.identifier, self.model.data_source.connstr)
	
	def update(self):
		
		if self.model.is_connected():
			self.set_title(os.path.split(str(self.model.identifier))[-1].strip("#"))
		self.descriptor_group.update()
		self.distance_group.update()
		self.cluster_group.update()
	
	def save(self):
		
		if self.model.data_source is None:
			self.dialogs.open("Connect")
			
		else:
			self.progress.show("Saving...")
			self.model.save()
			self.progress.reset()
	
	def set_clusters(self, clusters, nodes, edges, labels, positions = {}):
		
		if edges:
			self.graph_view.set_data(self.model.sample_data, clusters, nodes, edges, labels, positions)
		else:
			self.graph_view.set_data(self.model.sample_data)
		self.graph_view.reset_scene()
		self.cluster_group.update_clusters_found(len(clusters) if clusters else 0)
	
	def get_icon(self, name):

		path = os.path.join(os.path.dirname(res.__file__), name)
		if os.path.isfile(path):
			return QtGui.QIcon(path)
		path = os.path.join(os.path.dirname(deposit.__file__), "res", name)
		if os.path.isfile(path):
			return QtGui.QIcon(path)
		raise Exception("Could not load icon", name)
	
	def save_dendrogram(self, path):
		
		self.graph_view.save_pdf(path)
	
	def save_catalog(self, path, scale = 1/3, dpi = 600, line_width = 0.5):
		
		clusters, _, _, labels, _ = self.model.clustering.get_data()
		clusters = dict([(labels[cluster_node_id], [labels[sample_node_id] for sample_node_id in clusters[cluster_node_id]]) for cluster_node_id in clusters])
		save_catalog(path, self.model.sample_data, clusters, scale, dpi, line_width, progress = self.progress)
	
	@QtCore.Slot()
	def on_load_data(self):
		
		nodes = None
		if self.model.lap_descriptors is not None:
			self.set_clusters(*self.model.load_samples())
		
		self.descriptor_group.update()
		self.distance_group.update()
		self.cluster_group.update()
		self.cluster_group.update_n_clusters()
		
		self.update()
	
	@QtCore.Slot()
	def on_cluster_classes_changed(self):
		
		self.cluster_group.update()
	
	@QtCore.Slot()
	def on_cluster(self):
		
		self.cluster_group.update_clusters_found(None)
		max_clusters, limit = self.cluster_group.get_limits()
		self.set_clusters(*self.model.clustering.make(max_clusters, limit))
		self.update()
	
	@QtCore.Slot()
	def on_update_tree(self):
		
		self.set_clusters(*self.model.clustering.update())
		self.update()
	
	@QtCore.Slot()
	def on_add_cluster(self):
		
		self.graph_view.add_cluster()
	
	@QtCore.Slot()
	def on_calculate(self):
		
		self.model.calc_distance()
		self.update()
	
	@QtCore.Slot()
	def on_delete_distance(self):
		
		reply = QtWidgets.QMessageBox.question(self, "Delete Distances", "Delete distances from database?", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
		if reply == QtWidgets.QMessageBox.Yes:
			self.model.delete_distance()
			self.update()
	
	@QtCore.Slot()
	def on_delete_clusters(self):
		
		reply = QtWidgets.QMessageBox.question(self, "Delete Clusters", "Delete clusters from database?", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
		if reply == QtWidgets.QMessageBox.Yes:
			self.model.clustering.delete()
			labels = dict([(str(sample_id), sample_id) for sample_id in self.model.sample_ids])
			self.graph_view.set_data(self.model.sample_data)
			self.graph_view.reset_scene()
			self.update()
	
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
		self.update_mrud()
	
	def on_data_changed(self, *args):
		
		self.statusbar.message("")
	
	def on_saved(self, *args):
		
		self.statusbar.message("Database saved.")
	
	def on_save_failed(self, *args):
		
		self.statusbar.message("Saving failed!")
	
	def closeEvent(self, event):
		
		if not self.model.is_saved():
			reply = QtWidgets.QMessageBox.question(self, "Exit", "Save changes to database?", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel)
			if reply == QtWidgets.QMessageBox.Yes:
				self.save()
			elif reply == QtWidgets.QMessageBox.No:
				pass
			else:
				event.ignore()
				return
		
		self.model.on_close()
		QtWidgets.QMainWindow.closeEvent(self, event)
