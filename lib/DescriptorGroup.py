
from lib.Button import (Button)
from lib.Combo import (Combo)
from lib.fnc_drawing import get_lap_descriptors

from deposit.store.Conversions import (as_path)
from deposit.DModule import (DModule)
from deposit import Broadcasts

from PySide2 import (QtWidgets, QtCore, QtGui)

from natsort import natsorted
from copy import copy
import json

class DescriptorGroup(DModule, QtWidgets.QGroupBox):
	
	load_data = QtCore.Signal()
	
	def __init__(self, view):
		
		self.view = view
		self.model = view.model
		self.changed = False
		self.lap_descriptors = None
		
		DModule.__init__(self)
		QtWidgets.QGroupBox.__init__(self, "Class / Descriptors")
		
		self.setLayout(QtWidgets.QVBoxLayout())
		
		self.class_combo = Combo(self.on_sample_class_changed)
		
		self.form_frame = QtWidgets.QFrame()
		self.form_frame.setLayout(QtWidgets.QFormLayout())
		self.form_frame.layout().setContentsMargins(0, 0, 0, 0)
		
		self.form_frame.layout().addRow(QtWidgets.QLabel("Sample Class:"), self.class_combo)
		self.form_frame.layout().addRow(QtWidgets.QLabel("Sample Descriptors:"), None)
		
		self.load_data_button = Button("Load Data", self.on_load_data)
		self.load_descr_button = Button("Load Descriptors...", self.on_load_descriptors)
		button_frame = QtWidgets.QFrame()
		button_frame.setLayout(QtWidgets.QHBoxLayout())
		button_frame.layout().setContentsMargins(0, 0, 0, 0)
		button_frame.layout().addWidget(self.load_data_button)
		button_frame.layout().addWidget(self.load_descr_button)
		
		self.layout().addWidget(self.form_frame)
		self.layout().addWidget(button_frame)
		
		self.load_descriptors()
		self.update()
		
		self.connect_broadcast(Broadcasts.STORE_LOADED, self.on_store_changed)
		self.connect_broadcast(Broadcasts.STORE_DATA_SOURCE_CHANGED, self.on_store_changed)
	
	def load_descriptors(self, descriptors = None):
		
		prev = copy(self.lap_descriptors)
		if (descriptors is None) or (self.lap_descriptors is None):
			self.lap_descriptors = get_lap_descriptors(descriptors)
		
		for row in range(self.form_frame.layout().rowCount())[::-1]:
			if row > 1:
				self.form_frame.layout().removeRow(row)
		
		sample_cls = self.class_combo.get_value()
		if not sample_cls:
			self.lap_descriptors = None
		else:
			for name in [
				"Custom_Id",
				"Profile_Rim",
				"Profile_Bottom",
				"Profile_Radius",
				"Profile_Geometry",
				"Profile_Radius_Point",
				"Profile_Rim_Point",
				"Profile_Bottom_Point",
				"Profile_Left_Side",
			]:
				self.lap_descriptors[name][0] = sample_cls
		
		if self.lap_descriptors != prev:
			self.changed = True
		
		if self.lap_descriptors is not None:
			for name in self.lap_descriptors:
				self.form_frame.layout().addRow(QtWidgets.QLabel("   %s" % (name)), QtWidgets.QLabel(".".join(self.lap_descriptors[name])))
	
	def update(self):
		
		self.load_data_button.setEnabled(self.model.is_connected() and self.changed and (self.lap_descriptors is not None))
	
	def update_sample(self):
		
		if self.model.is_connected():
			self.class_combo.set_values(natsorted(list(self.model.classes.keys())), "Sample")
		else:
			self.class_combo.set_values([])
	
	def on_store_changed(self, *args):
		
		self.changed = True
		self.update_sample()
		self.load_descriptors()
		self.update()
	
	@QtCore.Slot()
	def on_sample_class_changed(self):
		
		self.changed = True
		self.load_descriptors()
		self.update()
	
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
