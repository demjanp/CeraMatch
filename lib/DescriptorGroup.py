from lib.Button import (Button)

from PySide2 import (QtWidgets, QtCore, QtGui)

from natsort import natsorted

class Combo(QtWidgets.QComboBox):
	
	def __init__(self, callback):
		
		QtWidgets.QComboBox.__init__(self)
		self.currentIndexChanged.connect(callback)
		
		self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)
	
	def clear_values(self):
		
		self.blockSignals(True)
		self.clear()
		self.blockSignals(False)
	
	def set_values(self, values, default = None):
		
		value = self.currentText()
		if value:
			default = value
		self.blockSignals(True)
		self.clear()
		self.addItems(values)
		if default in values:
			self.setCurrentIndex(values.index(default))
		self.blockSignals(False)
	
	def get_value(self):
		
		return self.currentText()

class DescriptorGroup(QtWidgets.QGroupBox):
	
	def __init__(self, view):
		
		self.view = view
		self.model = view.model
		self.changed = False
		
		QtWidgets.QGroupBox.__init__(self, "Class / Descriptors")
		
		self.setLayout(QtWidgets.QVBoxLayout())
		
		self.class_combo = Combo(self.on_changed)
		self.id_combo = Combo(self.on_changed)
		self.profile_combo = Combo(self.on_changed)
		self.radius_combo = Combo(self.on_changed)
		self.recons_combo = Combo(self.on_changed)
		
		self.load_button = Button("Load", self.on_load)
		
		form_frame = QtWidgets.QFrame()
		form_frame.setLayout(QtWidgets.QFormLayout())
		form_frame.layout().setContentsMargins(0, 0, 0, 0)
		
		cluster_label = QtWidgets.QLineEdit("CMCluster")
		cluster_label.setReadOnly(True)
		cluster_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)
		
		form_frame.layout().addRow(QtWidgets.QLabel("Sample Class:"), self.class_combo)
		form_frame.layout().addRow(QtWidgets.QLabel("Cluster Class:"), cluster_label)
		form_frame.layout().addRow(QtWidgets.QLabel("Sample Descriptors:"), None)
		form_frame.layout().addRow(QtWidgets.QLabel("Id:"), self.id_combo)
		form_frame.layout().addRow(QtWidgets.QLabel("Profile:"), self.profile_combo)
		form_frame.layout().addRow(QtWidgets.QLabel("Radius:"), self.radius_combo)
		form_frame.layout().addRow(QtWidgets.QLabel("Drawing:"), self.recons_combo)
		self.layout().addWidget(form_frame)
		self.layout().addWidget(self.load_button)
	
	def update(self):
		
		if self.model.is_connected():
			self.class_combo.set_values(natsorted(list(self.model.classes.keys())), "Sample")
			sample_cls = self.class_combo.get_value()
			if sample_cls:
				descriptors = self.model.classes[sample_cls].descriptors
				self.id_combo.set_values(descriptors, "Id")
				self.profile_combo.set_values(descriptors, "Profile")
				self.radius_combo.set_values(descriptors, "Radius")
				self.recons_combo.set_values(descriptors, "Reconstruction")
		
		self.load_button.setEnabled(self.model.is_connected() and self.changed and (self.get_values() is not None))
	
	def get_values(self):
		# return [Sample Class, ID Descriptor, Profile Descriptor, Radius Descriptor, Reconstruction Descriptor]
		
		values = []
		controls = [self.class_combo, self.id_combo, self.profile_combo, self.radius_combo, self.recons_combo]
		for control in controls:
			value = control.get_value()
			if value:
				values.append(value)
		if len(values) == len(controls):
			return values
		return None
	
	def on_changed(self, *args):
		
		self.changed = True
		self.update()
	
	def on_load(self, *args):
		
		self.changed = False
		self.view.reload_samples()
