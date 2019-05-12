from lib.Model import (Model)
from lib.ImageList import (ImageList)
from lib.WeightsFrame import (WeightsFrame)
from lib.ClusterFrame import (ClusterFrame)
from lib.FooterFrame import (FooterFrame)
from lib.StatusBar import (StatusBar)

from PySide2 import (QtWidgets, QtCore, QtGui)
import os

class View(QtWidgets.QMainWindow):

	def __init__(self):
		
		self.model = None
		self.registry_dc = None
		self._last_selected = None
		self._loaded = False
		
		QtWidgets.QMainWindow.__init__(self)
		
		self.model = Model(self)
		
		self.setWindowTitle("CeraMatch")
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
		self.cluster_frame = ClusterFrame(self)
		self.statusbar = StatusBar(self)
		
		self.samples_button = QtWidgets.QPushButton("Sort by Sample IDs")
		self.samples_button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
		self.samples_button.clicked.connect(self.on_samples)
		
		self.distance_button = QtWidgets.QPushButton("Sort by Distance")
		self.distance_button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
		self.distance_button.clicked.connect(self.on_distance)
		
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
		self.left_frame.layout().addWidget(self.cluster_frame)
		self.left_frame.layout().addStretch()
		
		self.right_frame.layout().addWidget(self.image_lst)
		self.right_frame.layout().addWidget(self.footer_frame)
		
		self.setStatusBar(self.statusbar)
		
		self.splitter.setStretchFactor(0,1)
		self.splitter.setStretchFactor(1,3)
		
		self._loaded = True
		
		self.setGeometry(100, 100, 1024, 768)
		
		self.footer_frame.slider_zoom.setValue(100)
		
		self.update()
	
	def get_selected(self):
		# returns [[sample_id, DResource, label, value, index], ...]
		
		return self.image_lst.get_selected()
	
	def update(self):
		
		if not self._loaded:
			return
		self.cluster_frame.update()
		self.weights_frame.update()
		
		selected = self.get_selected()
		
		self.distance_button.setEnabled((len(selected) > 0) or (self._last_selected is not None))
		
		if selected:
			self.statusbar.message("Label: %s, Sample ID: %s" % (selected[0].value, selected[0].id))
	
	def on_slider(self, name, value):
		
		if name in self.model.weights:
			self.model.set_weight(name, value / 100)
			return
	
	def on_samples(self, *args):
		
		self.model.load_ids()
	
	def on_distance(self, *args):
		
		selected = self.image_lst.get_selected()
		if selected:
			self._last_selected = selected[0].id
		if self._last_selected is None:
			return
		self.model.load_distance(self._last_selected)
	
	def on_zoom(self, value):
		
		self.image_lst.set_thumbnail_size(value)

