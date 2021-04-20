
from lib.Button import (Button)
from lib.Combo import (Combo)
#from lib.fnc_drawing import get_lap_descriptors

from deposit.store.Conversions import (as_path)
from deposit.DModule import (DModule)
from deposit import Broadcasts

from PySide2 import (QtWidgets, QtCore, QtGui)

from natsort import natsorted
from copy import copy
import json

class HeaderButton(QtWidgets.QToolButton):
	
	def __init__(self, label):
		
		QtWidgets.QToolButton.__init__(self)
		
		self.setText(label)
		self.setArrowType(QtCore.Qt.RightArrow)
		self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
		self.setAutoRaise(True)
		self.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
		self.setStyleSheet("QToolButton {font: bold; border-top: 1px solid white; border-left: 1px solid white; border-bottom: 1px solid gray; border-right: 1px solid gray;} QToolButton:pressed {border-top: 1px solid gray; border-left: 1px solid gray; border-bottom: 1px solid white; border-right: 1px solid white;}")

class DescriptorGroup(DModule, QtWidgets.QGroupBox):
	
	load_data = QtCore.Signal()
	cluster_classes_changed = QtCore.Signal()
	
	def __init__(self, view):
		
		self.view = view
		self.model = view.model
		self.changed = False
		
		DModule.__init__(self)
		QtWidgets.QGroupBox.__init__(self, "Class / Descriptors")
		
		self.setLayout(QtWidgets.QVBoxLayout())
		
		self.sample_class_combo = Combo(self.on_sample_class_changed)
		self.cluster_class_combo = Combo(self.on_cluster_class_changed, editable = True)
		self.node_class_combo = Combo(self.on_node_class_changed, editable = True)
		
		classes_frame = QtWidgets.QFrame()
		classes_frame.setLayout(QtWidgets.QFormLayout())
		classes_frame.layout().setContentsMargins(0, 0, 0, 0)
		
		classes_frame.layout().addRow(QtWidgets.QLabel("Sample Class:"), self.sample_class_combo)
		classes_frame.layout().addRow(QtWidgets.QLabel("Cluster Class:"), self.cluster_class_combo)
		classes_frame.layout().addRow(QtWidgets.QLabel("Node Class:"), self.node_class_combo)
		
		self.descriptors_frame = QtWidgets.QFrame()
		self.descriptors_frame.setLayout(QtWidgets.QFormLayout())
		self.descriptors_frame.layout().setContentsMargins(0, 0, 0, 0)
		self.descriptors_frame.setVisible(False)
		
		self.load_data_button = Button("Load Data", self.on_load_data)
		self.load_descr_button = Button("Load Descriptors...", self.on_load_descriptors)
		button_frame = QtWidgets.QFrame()
		button_frame.setLayout(QtWidgets.QHBoxLayout())
		button_frame.layout().setContentsMargins(0, 0, 0, 0)
		button_frame.layout().addWidget(self.load_data_button)
		button_frame.layout().addWidget(self.load_descr_button)
		
		descriptor_box = QtWidgets.QFrame()
		descriptor_box.setLayout(QtWidgets.QVBoxLayout())
		descriptor_box.layout().setContentsMargins(0, 0, 0, 0)
		self.descriptor_header_button = HeaderButton("Descriptors")
		descriptor_box.layout().addWidget(self.descriptor_header_button)
		descriptor_box.layout().addWidget(self.descriptors_frame)
		self.descriptor_header_button.clicked.connect(self.on_descriptors_clicked)
		
		self.layout().addWidget(classes_frame)
		self.layout().addWidget(descriptor_box)
		self.layout().addWidget(button_frame)
		
		self.load_descriptors()
		self.update()
		
		self.connect_broadcast(Broadcasts.STORE_LOADED, self.on_store_changed)
		self.connect_broadcast(Broadcasts.STORE_DATA_SOURCE_CHANGED, self.on_store_changed)
		self.connect_broadcast(Broadcasts.STORE_DATA_CHANGED, self.on_store_changed)
	
	def load_descriptors(self, descriptors = None):
		
		prev = copy(self.model.lap_descriptors)
		
		sample_cls = self.sample_class_combo.get_value()
		self.model.load_lap_descriptors(descriptors, sample_cls)
		
		for row in range(self.descriptors_frame.layout().rowCount())[::-1]:
			self.descriptors_frame.layout().removeRow(row)
		
		if self.model.lap_descriptors != prev:
			self.changed = True
		
		if self.model.lap_descriptors is not None:
			for name in self.model.lap_descriptors:
				self.descriptors_frame.layout().addRow(QtWidgets.QLabel("   %s" % (name)), QtWidgets.QLabel(".".join(self.model.lap_descriptors[name])))
	
	def update(self):
		
		self.load_data_button.setEnabled(self.model.is_connected() and self.changed and (self.model.lap_descriptors is not None))
	
	def update_classes(self):
		
		if self.model.is_connected():
			descriptors = set(self.model.descriptor_names)
			classes = natsorted([name for name in self.model.classes if name not in descriptors])
			self.sample_class_combo.set_values(classes, "Sample")
			self.cluster_class_combo.set_values(classes, "CMCluster")
			self.node_class_combo.set_values(classes, "CMNode")
		else:
			self.sample_class_combo.set_values([])
			self.cluster_class_combo.set_values([])
			self.node_class_combo.set_values([])
		self.model.cluster_class = self.cluster_class_combo.get_value()
		self.model.node_class = self.node_class_combo.get_value()

	def on_store_changed(self, *args):
		
		self.changed = True
		self.update_classes()
		self.load_descriptors()
		self.update()
	
	@QtCore.Slot()
	def on_descriptors_clicked(self):
		
		visible = self.descriptors_frame.isVisible()
		if visible:
			self.descriptor_header_button.setArrowType(QtCore.Qt.RightArrow)
		else:
			self.descriptor_header_button.setArrowType(QtCore.Qt.DownArrow)
		self.descriptors_frame.setVisible(not visible)
	
	@QtCore.Slot()
	def on_sample_class_changed(self):
		
		self.changed = True
		self.load_descriptors()
		self.update()
		self.cluster_classes_changed.emit()
	
	@QtCore.Slot()
	def on_node_class_changed(self):
		
		self.changed = True
		self.model.node_class = self.node_class_combo.get_value()
		self.update()
		self.cluster_classes_changed.emit()
	
	@QtCore.Slot()
	def on_cluster_class_changed(self):
		
		self.changed = True
		self.model.cluster_class = self.cluster_class_combo.get_value()
		self.update()
		self.cluster_classes_changed.emit()
	
	@QtCore.Slot()
	def on_load_data(self):
		
		self.changed = False
		self.load_data.emit()
	
	@QtCore.Slot()
	def on_load_descriptors(self):
		
		url, format = QtWidgets.QFileDialog.getOpenFileUrl(self.view, caption = "Import Descriptors", filter = "(*.txt)")
		url = str(url.toString())
		if not url:
			return
		path = as_path(url)
		if path is None:
			return
		with open(path, "r") as f:
			_, descriptors = json.load(f)
		self.load_descriptors(descriptors)
		self.update()
