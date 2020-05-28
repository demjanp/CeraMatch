from lib.Model import (Model)
from lib.ImageView import (ImageView)  # DEBUG
from lib.ClusterGroup import (ClusterGroup)
from lib.FooterFrame import (FooterFrame)
from lib.Menu import (Menu)
from lib.ToolBar import (ToolBar)
from lib.StatusBar import (StatusBar)
from lib.Button import (Button)

from deposit.DModule import (DModule)
from deposit import Broadcasts

from PySide2 import (QtWidgets, QtCore, QtGui)

class View(DModule, QtWidgets.QMainWindow):
	
	MODE_IDS = 0x00000001
	MODE_DISTANCE_MIN = 0x00000002
	MODE_DISTANCE_MAX = 0x00000004
	MODE_CLUSTER = 0x00000008
	
	def __init__(self):
		
		self.model = None
		self.registry_dc = None
		self.mode = None
		self._loaded = False
		
		DModule.__init__(self)
		QtWidgets.QMainWindow.__init__(self)
		
		self.model = Model(self)
		
		self.setWindowTitle("CeraMatch")
		self.setWindowIcon(QtGui.QIcon("res\cm_icon.svg"))
		self.setStyleSheet("QPushButton {padding: 5px; min-width: 100px;}")
		
		self.central_widget = QtWidgets.QWidget(self)
		self.central_widget.setLayout(QtWidgets.QVBoxLayout())
		self.central_widget.layout().setContentsMargins(0, 0, 0, 0)
		self.setCentralWidget(self.central_widget)
		
		self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
		
		self.central_widget.layout().addWidget(self.splitter)
		
		self.image_view = ImageView(self)
		self.footer_frame = FooterFrame(self)
		self.cluster_group = ClusterGroup(self)
		self.menu = Menu(self)
		self.toolbar = ToolBar(self)
		self.statusbar = StatusBar(self)
		
		self.calculate_button = Button("Calculate Distances", self.on_calculate)
		self.calculate_button.setEnabled(False)
		self.samples_button = Button("Sort by Sample IDs", self.on_samples)
		self.samples_button.setEnabled(False)
		self.distance_max_button = Button("Sort by Max Distance", self.on_distance_max)
		self.distance_max_button.setEnabled(False)
		self.distance_min_button = Button("Sort by Min Distance", self.on_distance_min)
		self.distance_min_button.setEnabled(False)
		self.cluster_button = Button("Sort by Clustering", self.on_cluster)
		self.cluster_button.setEnabled(False)
		
		self.left_frame = QtWidgets.QFrame(self)
		self.left_frame.setLayout(QtWidgets.QVBoxLayout())
		self.left_frame.layout().setContentsMargins(10, 10, 0, 10)
		
		self.right_frame = QtWidgets.QFrame(self)
		self.right_frame.setLayout(QtWidgets.QVBoxLayout())
		self.right_frame.layout().setContentsMargins(0, 0, 0, 0)
		
		self.splitter.addWidget(self.left_frame)
		self.splitter.addWidget(self.right_frame)
		
		self.left_frame.layout().addWidget(self.cluster_group)
		self.left_frame.layout().addWidget(self.calculate_button)
		self.left_frame.layout().addWidget(self.samples_button)
		self.left_frame.layout().addWidget(self.distance_max_button)
		self.left_frame.layout().addWidget(self.distance_min_button)
		self.left_frame.layout().addWidget(self.cluster_button)
		self.left_frame.layout().addWidget(self.cluster_group)
		self.left_frame.layout().addStretch()
		
		self.right_frame.layout().addWidget(self.image_view)
		self.right_frame.layout().addWidget(self.footer_frame)
		
		self.setStatusBar(self.statusbar)
		
		self._loaded = True
		
		self.setGeometry(100, 100, 1024, 768)
		
		self.on_samples()
		
		self.footer_frame.slider_zoom.setValue(100)
		
		self.connect_broadcast(Broadcasts.VIEW_ACTION, self.on_update)
		self.connect_broadcast(Broadcasts.STORE_LOCAL_FOLDER_CHANGED, self.on_update)
		self.connect_broadcast(Broadcasts.STORE_DATA_SOURCE_CHANGED, self.on_update)
		self.connect_broadcast(Broadcasts.STORE_DATA_CHANGED, self.on_update)
		self.connect_broadcast(Broadcasts.STORE_SAVED, self.on_update)
		self.connect_broadcast(Broadcasts.STORE_LOADED, self.on_loaded)
	
	def get_selected(self):
		# returns [[sample_id, DResource, label, value, index], ...]
		
		return self.image_view.get_selected()
	
	def update(self):
		
		if not hasattr(self, "cluster_group"):
			return

		self.cluster_group.update()
		self.footer_frame.update()
		self.image_view.update_()
		
		selected = self.get_selected()
		
		self.calculate_button.setEnabled(self.model.is_connected() and not self.model.has_distances())
		self.samples_button.setEnabled(self.model.is_connected())
		self.distance_max_button.setEnabled(self.model.has_distances())
		self.distance_min_button.setEnabled(self.model.has_distances() and ((len(selected) > 0) or ((len(self.model.samples) > 0) and isinstance(self.model.samples[0].value, float))))
		self.cluster_button.setEnabled(self.model.has_clusters())
		
		if selected:
			cluster = selected[0].cluster
			levels = self.image_view.get_selected_level()
			if levels:
				cluster = ".".join(cluster.split(".")[:levels[0]])
			
			if cluster:
				text = "Cluster: %s, Label: %s, Sample ID: %s" % (cluster, selected[0].value, selected[0].id)
			else:
				text = "Label: %s, Sample ID: %s" % (selected[0].value, selected[0].id)
			self.statusbar.message(text)
	
	def on_update(self, *args):
		
		self.update()
	
	def on_loaded(self, *args):
		
		self.model.load_samples()
		self.update()
		self.on_samples()
	
	def on_calculate(self, *args):
		
		self.model.calc_distances()
	
	def on_samples(self, *args):
		
		self.mode = self.MODE_IDS
		self.model.sort_by_ids()
	
	def on_distance_max(self, *args):
		
		self.mode = self.MODE_DISTANCE_MAX
		self.model.sort_by_distmax()
	
	def on_distance_min(self, *args):
		
		selected = self.get_selected()
		if selected:
			sample_id = selected[0].id
		elif isinstance(self.model.samples[0].value, float):
			sample_id = self.model.samples[0].id
		else:
			return
		self.mode = self.MODE_DISTANCE_MIN
		self.model.sort_by_distance(sample_id)
	
	def on_cluster(self, *args):
		
		self.mode = self.MODE_CLUSTER
		
		if self.model.has_clusters():
			self.model.populate_clusters()
			self.model.sort_by_cluster()
	
	def on_auto_cluster(self, *args):
		
		self.mode = self.MODE_CLUSTER
		self.model.auto_cluster()
		self.model.populate_clusters()
	
	def on_split_cluster(self, *args):
		
		selected = self.image_view.get_selected()
		if len(selected) != 1:
			return
		cluster = selected[0].cluster
		if cluster is None:
			return
		self.model.split_cluster(cluster)
		self.model.populate_clusters(selected[0])
	
	def on_join_parent(self, *args):
		
		selected = self.image_view.get_selected()
		if not selected:
			return
		self.mode = self.MODE_CLUSTER

		clusters = set()
		for sample in selected:
			if sample.cluster is None:
				continue
			clusters.add(sample.cluster)
		if clusters:
			for cluster in clusters:
				self.model.join_cluster_to_parent(cluster)
		else:
			self.model.join_cluster_to_parent()
		self.model.populate_clusters(selected[0])
	
	def on_join_children(self, *args):
		
		selected = self.image_view.get_selected()
		if not selected:
			return
		self.mode = self.MODE_CLUSTER

		clusters = set()
		for sample in selected:
			if sample.cluster is None:
				continue
			clusters.add(sample.cluster)
		if clusters:
			
			levels = self.image_view.get_selected_level()
			if not levels:
				return
			level = max(levels)
			for cluster in clusters:
				self.model.join_children_to_cluster(cluster, level)
		else:
			return
		self.model.populate_clusters(selected[0])
	
	def on_manual_cluster(self, *args):
		
		selected = self.image_view.get_selected()
		if not selected:
			return
		self.model.manual_cluster(selected)
		self.model.populate_clusters(selected[0])
	
	def on_clear_clusters(self, *args):
		
		self.model.clear_clusters()
		self.model.sort_by_ids()
		self.model.sort_by_cluster()
	
	def on_reload(self, *args):
		
		if self.mode is None:
			return
		if self.mode == self.MODE_IDS:
			self.on_samples()
		elif self.mode == self.MODE_DISTANCE_MIN:
			self.on_distance_min()
		elif self.mode == self.MODE_DISTANCE_MAX:
			self.on_distance_max()
		elif self.mode == self.MODE_CLUSTER:
			self.on_cluster()
	
	def on_prev(self, *args):
		
		self.model.browse_distmax(-1)
	
	def on_next(self, *args):
		
		self.model.browse_distmax(1)
	
	def on_zoom(self, value):
		
		self.image_view.set_thumbnail_size(value)
	
	def closeEvent(self, event):
		
		if not self.model.is_saved():
			reply = QtWidgets.QMessageBox.question(self, "Exit", "Save changes to database?", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel)
			if reply == QtWidgets.QMessageBox.Yes:
				self.toolbar.on_save()
			elif reply == QtWidgets.QMessageBox.No:
				pass
			else:
				event.ignore()
				return
		
		self.model.on_close()

