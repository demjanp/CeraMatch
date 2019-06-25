from lib.Model import (Model)
from lib.ImageList import (ImageList)
from lib.WeightsFrame import (WeightsFrame)
from lib.ClusterGroup import (ClusterGroup)
from lib.FooterFrame import (FooterFrame)
from lib.ToolBar import (ToolBar)
from lib.StatusBar import (StatusBar)
from lib.Button import (Button)

from PySide2 import (QtWidgets, QtCore, QtGui)
import os

class View(QtWidgets.QMainWindow):
	
	MODE_IDS = 100
	MODE_DISTANCE_MIN = 200
	MODE_DISTANCE_MAX = 300
	MODE_CLUSTER = 400
	
	def __init__(self):
		
		self.model = None
		self.registry_dc = None
		self.mode = None
		self._loaded = False
		
		QtWidgets.QMainWindow.__init__(self)
		
		self.model = Model(self)
		
		self.setWindowTitle("CeraMatch")
		self.setWindowIcon(QtGui.QIcon("res\cm_icon.svg"))
		self.setStyleSheet("QWidget { font-size: 11pt;} QPushButton {font-size: 11pt; padding: 5px; min-width: 100px;}")
		
		self.central_widget = QtWidgets.QWidget(self)
		self.central_widget.setLayout(QtWidgets.QVBoxLayout())
		self.central_widget.layout().setContentsMargins(0, 0, 0, 0)
		self.setCentralWidget(self.central_widget)
		
		self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
		self.splitter.setStyleSheet("QSplitter::handle:horizontal { image: url(res/handle.png); margin: 2px;}")
		
		self.central_widget.layout().addWidget(self.splitter)
		
		self.image_lst = ImageList(self)
		self.footer_frame = FooterFrame(self)
		self.weights_frame = WeightsFrame(self)
		self.cluster_group = ClusterGroup(self)
		self.toolbar = ToolBar(self)
		self.statusbar = StatusBar(self)
		
		self.samples_button = Button("Sort by Sample IDs", self.on_samples)
		self.distance_max_button = Button("Sort by Max Distance", self.on_distance_max)
		self.distance_min_button = Button("Sort by Min Distance", self.on_distance_min)
		self.cluster_button = Button("Sort by Clustering", self.on_cluster)
		
		self.left_frame = QtWidgets.QFrame(self)
		self.left_frame.setLayout(QtWidgets.QVBoxLayout())
		self.left_frame.layout().setContentsMargins(10, 10, 0, 10)
		
		self.right_frame = QtWidgets.QFrame(self)
		self.right_frame.setLayout(QtWidgets.QVBoxLayout())
		self.right_frame.layout().setContentsMargins(0, 0, 0, 0)
		
		self.splitter.addWidget(self.left_frame)
		self.splitter.addWidget(self.right_frame)
		
		self.left_frame.layout().addWidget(self.weights_frame)
		self.left_frame.layout().addWidget(self.cluster_group)
		self.left_frame.layout().addWidget(self.samples_button)
		self.left_frame.layout().addWidget(self.distance_max_button)
		self.left_frame.layout().addWidget(self.distance_min_button)
		self.left_frame.layout().addWidget(self.cluster_button)
		self.left_frame.layout().addWidget(self.cluster_group)
		self.left_frame.layout().addStretch()
		
		self.right_frame.layout().addWidget(self.image_lst)
		self.right_frame.layout().addWidget(self.footer_frame)
		
		self.setStatusBar(self.statusbar)
		
		self.splitter.setStretchFactor(0,1)
		self.splitter.setStretchFactor(1,3)
		
		self._loaded = True
		
		self.setGeometry(100, 100, 1024, 768)
		
		self.footer_frame.slider_zoom.setValue(100)
		
		self.on_samples()
	
	def get_selected(self):
		# returns [[sample_id, DResource, label, value, index], ...]
		
		return self.image_lst.get_selected()
	
	def update(self):
		
		if not self._loaded:
			return
		
		if not self.model.samples:
			return
		
		self.weights_frame.update()
		self.cluster_group.update()
		self.footer_frame.update()
		self.image_lst.update_()
		
		selected = self.get_selected()
		
		self.distance_min_button.setEnabled((len(selected) > 0) or isinstance(self.model.samples[0].value, float))
		
		if selected:
			cluster = selected[0].cluster
			if cluster:
				text = "Cluster: %s, Label: %s, Sample ID: %s" % (cluster, selected[0].value, selected[0].id)
			else:
				text = "Label: %s, Sample ID: %s" % (selected[0].value, selected[0].id)
			self.statusbar.message(text)
	
	def on_slider(self, name, value):
		
		if name in self.model.weights:
			self.model.set_weight(name, value / 100)
			return
	
	def on_load_weights(self, *args):
		
		selected = self.image_lst.get_selected()
		if not selected:
			return
		cluster = selected[0].cluster
		if cluster in self.model.cluster_weights:
			self.model.set_weights(self.model.cluster_weights[cluster])
	
	def on_samples(self, *args):
		
		self.mode = self.MODE_IDS
		self.model.load_ids()
	
	def on_distance_max(self, *args):
		
		self.mode = self.MODE_DISTANCE_MAX
		self.model.load_distmax()
	
	def on_distance_min(self, *args):
		
		selected = self.get_selected()
		if selected:
			sample_id = selected[0].id
		elif isinstance(self.model.samples[0].value, float):
			sample_id = self.model.samples[0].id
		else:
			return
		self.mode = self.MODE_DISTANCE_MIN
		self.model.load_distance(sample_id)
	
	def on_cluster(self, *args):
		
		self.mode = self.MODE_CLUSTER
		
		if self.model.has_clusters():
			self.model.populate_clusters()
		else:
			self.model.sort_by_leaf()
	
	def on_split_cluster(self, *args):
		
		selected = self.image_lst.get_selected()
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
				self.model.split_cluster(cluster)
		else:
			self.model.split_cluster()
		self.model.populate_clusters(selected[0])
	
	def on_join_cluster(self, *args):
		
		selected = self.image_lst.get_selected()
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
				self.model.join_cluster(cluster)
		else:
			self.model.join_cluster()
		self.model.populate_clusters(selected[0])
	
	def on_manual_cluster(self, *args):
		
		selected = self.image_lst.get_selected()
		if not selected:
			return
		self.model.manual_cluster(selected)
		self.model.populate_clusters(selected[0])
	
	def on_split_all_clusters(self, *args):
		
		self.mode = self.MODE_CLUSTER
		self.model.split_all_clusters()
		self.model.populate_clusters()
	
	def on_set_outlier(self, *args):
		
		selected = self.image_lst.get_selected()
		if not selected:
			return
		self.model.set_outlier(selected)
	
	def on_set_central(self, *args):
		
		selected = self.image_lst.get_selected()
		if not selected:
			return
		self.model.set_central(selected)
	
	def on_clear_clusters(self, *args):
		
		self.model.clear_clusters()
		self.model.load_ids()
		self.model.sort_by_leaf()
	
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
		
		self.image_lst.set_thumbnail_size(value)

