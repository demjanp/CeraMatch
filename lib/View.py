from lib.Model import (Model)
from lib.ImageList import (ImageList)
from lib.WeightsFrame import (WeightsFrame)
from lib.FooterFrame import (FooterFrame)
from lib.ToolBar import (ToolBar)
from lib.StatusBar import (StatusBar)
from lib.Button import (Button)

from PySide2 import (QtWidgets, QtCore, QtGui)
import os

class View(QtWidgets.QMainWindow):
	
	MODE_IDS = 100
	MODE_DISTANCE = 200
	MODE_CLUSTER = 300
	
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
		self.toolbar = ToolBar(self)
		self.statusbar = StatusBar(self)
		
		self.samples_button = Button("Sort by Sample IDs", self.on_samples)
		self.distance_button = Button("Sort by Distance", self.on_distance)
		self.cluster_button = Button("Sort by Cluster", self.on_cluster)
		self.split_cluster_button = Button("Split Cluster", self.on_split_cluster)
		self.join_cluster_button = Button("Join Cluster", self.on_join_cluster)
		
		self.left_frame = QtWidgets.QFrame(self)
		self.left_frame.setLayout(QtWidgets.QVBoxLayout())
		self.left_frame.layout().setContentsMargins(10, 10, 0, 10)
		
		self.right_frame = QtWidgets.QFrame(self)
		self.right_frame.setLayout(QtWidgets.QVBoxLayout())
		self.right_frame.layout().setContentsMargins(0, 0, 0, 0)
		
		self.splitter.addWidget(self.left_frame)
		self.splitter.addWidget(self.right_frame)
		
		self.left_frame.layout().addWidget(self.weights_frame)
		self.left_frame.layout().addWidget(self.samples_button)
		self.left_frame.layout().addWidget(self.distance_button)
		self.left_frame.layout().addWidget(self.cluster_button)
		self.left_frame.layout().addWidget(self.split_cluster_button)
		self.left_frame.layout().addWidget(self.join_cluster_button)
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
		self.weights_frame.update()
		self.footer_frame.update()
		self.image_lst.update_()
		
		selected = self.get_selected()
		
		self.split_cluster_button.setEnabled(len(selected) > 0)
		
		self.join_cluster_button.setEnabled((len(selected) > 0) and (selected[0].has_cluster()))
		
		if selected:
			cluster = selected[0].cluster
			if cluster:
				text = "Label: %s, Cluster: %s, Sample ID: %s" % (selected[0].value, cluster, selected[0].id)
			else:
				text = "Label: %s, Sample ID: %s" % (selected[0].value, selected[0].id)
			self.statusbar.message(text)
			
			if cluster in self.model.cluster_weights:
				self.model.set_weights(self.model.cluster_weights[cluster])
	
	def on_slider(self, name, value):
		
		if name in self.model.weights:
			self.model.set_weight(name, value / 100)
			return
	
	def on_samples(self, *args):
		
		self.mode = self.MODE_IDS
		self.model.load_ids()
	
	def on_distance(self, *args):
		
		self.mode = self.MODE_DISTANCE
		selected = self.get_selected()
		if selected or isinstance(self.model.samples[0].value, float):
			if selected:
				self.model.load_distance(selected[0].id)
			else:
				self.model.load_distance(self.model.samples[0].id)
		else:
			self.model.load_distmax()
	
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
		self.model.split_cluster(selected[0])
	
	def on_join_cluster(self, *args):
		
		selected = self.image_lst.get_selected()
		if not selected:
			return
		self.mode = self.MODE_CLUSTER
		self.model.join_cluster(selected[0])
	
	def on_reload(self, *args):
		
		if self.mode is None:
			return
		if self.mode == self.MODE_IDS:
			self.on_samples()
		elif self.mode == self.MODE_DISTANCE:
			self.on_distance()
		elif self.mode == self.MODE_CLUSTER:
			self.on_cluster()
	
	def on_prev(self, *args):
		
		self.model.browse_distmax(-1)
	
	def on_next(self, *args):
		
		self.model.browse_distmax(1)
	
	def on_zoom(self, value):
		
		self.image_lst.set_thumbnail_size(value)

