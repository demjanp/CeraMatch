
from ceramatch.view.vcontrols_groups.controls.button import (Button)
from ceramatch.view.vcontrols_groups.controls.combo import (Combo)

from PySide2 import (QtWidgets, QtCore, QtGui)

class HeaderButton(QtWidgets.QToolButton):
	
	def __init__(self, label):
		
		QtWidgets.QToolButton.__init__(self)
		
		self.setText(label)
		self.setArrowType(QtCore.Qt.RightArrow)
		self.setSizePolicy(
			QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed
		)
		self.setAutoRaise(True)
		self.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
		self.setStyleSheet('''
			QToolButton {
				border-top: 1px solid white; 
				border-left: 1px solid white; border-bottom: 1px solid gray; 
				border-right: 1px solid gray;
			} 
			QToolButton:pressed {
				border-top: 1px solid gray; border-left: 1px solid gray; 
				border-bottom: 1px solid white; border-right: 1px solid white;
			}
		''')

class Descriptors(QtWidgets.QGroupBox):
	
	signal_load_drawings = QtCore.Signal()
	signal_cluster_classes_changed = QtCore.Signal()
	
	def __init__(self):
		
		QtWidgets.QGroupBox.__init__(self, "Classes / Descriptors")
		
		self.setStyleSheet("QGroupBox {font-weight: bold;}")
		self.setLayout(QtWidgets.QVBoxLayout())
		
		self.cluster_class_combo = Combo(
			self.on_cluster_class_changed, editable = True
		)
		self.node_class_combo = Combo(
			self.on_node_class_changed, editable = True
		)
		self.position_class_combo = Combo(
			self.on_position_class_changed, editable = True
		)
		
		classes_frame = QtWidgets.QFrame()
		classes_frame.setLayout(QtWidgets.QFormLayout())
		classes_frame.layout().setContentsMargins(0, 0, 0, 0)
		
		classes_frame.layout().addRow(
			QtWidgets.QLabel("Cluster Class:"), self.cluster_class_combo
		)
		classes_frame.layout().addRow(
			QtWidgets.QLabel("Node Class:"), self.node_class_combo
		)
		classes_frame.layout().addRow(
			QtWidgets.QLabel("Position Descriptor:"), self.position_class_combo
		)
		
		self.descriptors_frame = QtWidgets.QFrame()
		self.descriptors_frame.setLayout(QtWidgets.QVBoxLayout())
		self.descriptors_frame.layout().setContentsMargins(0, 0, 0, 0)
		
		self.descriptors_form = QtWidgets.QFrame()
		self.descriptors_form.setLayout(QtWidgets.QFormLayout())
		self.descriptors_form.layout().setContentsMargins(0, 0, 0, 0)
		
		self.descriptors_frame.layout().addWidget(self.descriptors_form)
		
		self.load_drawings_button = Button("Load Drawings", self.on_load_drawings)
		
		descriptor_box = QtWidgets.QFrame()
		descriptor_box.setLayout(QtWidgets.QVBoxLayout())
		descriptor_box.layout().setContentsMargins(0, 0, 0, 0)
		self.descriptor_header_button = HeaderButton("Descriptors")
		descriptor_box.layout().addWidget(self.descriptor_header_button)
		descriptor_box.layout().addWidget(self.descriptors_frame)
		self.descriptor_header_button.clicked.connect(
			self.on_descriptors_clicked
		)
		
		self.layout().addWidget(classes_frame)
		self.layout().addWidget(descriptor_box)
		self.layout().addWidget(self.load_drawings_button)
		
		self.descriptors_frame.setVisible(False)
	
	
	# ---- Signal handling
	# ------------------------------------------------------------------------
	@QtCore.Slot()
	def on_descriptors_clicked(self):
		
		visible = self.descriptors_frame.isVisible()
		if visible:
			self.descriptor_header_button.setArrowType(QtCore.Qt.RightArrow)
		else:
			self.descriptor_header_button.setArrowType(QtCore.Qt.DownArrow)
		self.descriptors_frame.setVisible(not visible)
	
	@QtCore.Slot()
	def on_node_class_changed(self):
		
		self.signal_cluster_classes_changed.emit()
	
	@QtCore.Slot()
	def on_position_class_changed(self):
		
		self.signal_cluster_classes_changed.emit()
	
	@QtCore.Slot()
	def on_cluster_class_changed(self):
		
		self.signal_cluster_classes_changed.emit()
	
	@QtCore.Slot()
	def on_load_drawings(self):
		
		self.signal_load_drawings.emit()
	
	
	# ---- get/set
	# ------------------------------------------------------------------------
	def clear(self):
		
		self.cluster_class_combo.clear_values()
		self.node_class_combo.clear_values()
		self.position_class_combo.clear_values()
		self.set_descriptors({})
	
	
	def set_cluster_class(self, name, classes):
		
		self.cluster_class_combo.set_values(classes, name)
	
	def set_node_class(self, name, classes):
		
		self.node_class_combo.set_values(classes, name)
	
	def set_position_class(self, name, classes):
		
		self.position_class_combo.set_values(classes, name)
	
	def set_descriptors(self, descriptors):
		# descriptors = {name: chain, ...}
		
		for row in range(self.descriptors_form.layout().rowCount())[::-1]:
			self.descriptors_form.layout().removeRow(row)
		for name in descriptors:
			self.descriptors_form.layout().addRow(
				QtWidgets.QLabel("   %s" % (name)), 
				QtWidgets.QLabel(descriptors[name]),
			)
	
	def set_load_drawings_enabled(self, state):
		
		self.load_drawings_button.setEnabled(state)
	
	def get_cluster_class(self):
		
		return self.cluster_class_combo.get_value()
	
	def get_node_class(self):
		
		return self.node_class_combo.get_value()
	
	def get_position_class(self):
		
		return self.position_class_combo.get_value()

