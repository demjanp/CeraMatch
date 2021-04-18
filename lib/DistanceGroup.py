
from lib.Button import (Button)

from deposit.DModule import (DModule)
from deposit import Broadcasts

from PySide2 import (QtWidgets, QtCore, QtGui)

class DistanceGroup(DModule, QtWidgets.QGroupBox):
	
	calculate = QtCore.Signal()
	delete = QtCore.Signal()
	
	def __init__(self, view):
		
		self.view = view
		self.model = view.model
		
		DModule.__init__(self)
		QtWidgets.QGroupBox.__init__(self, "Distance")
		
		self.setLayout(QtWidgets.QVBoxLayout())
		
		self.calculate_button = Button("Calculate Distances", self.on_calculate)
		self.delete_button = Button("Delete Distances", self.on_delete)
		
		self.layout().addWidget(self.calculate_button)
		self.layout().addWidget(self.delete_button)
		
		self.update()
		
		self.connect_broadcast(Broadcasts.STORE_LOADED, self.on_data_source_changed)
		self.connect_broadcast(Broadcasts.STORE_DATA_SOURCE_CHANGED, self.on_data_source_changed)
		self.connect_broadcast(Broadcasts.STORE_DATA_CHANGED, self.on_data_changed)
	
	def update(self):
		
		self.calculate_button.setEnabled(self.model.has_samples() and (not self.model.has_distance()))
		self.delete_button.setEnabled(self.model.has_samples() and self.model.has_distance())
	
	@QtCore.Slot()
	def on_calculate(self):
		
		self.calculate.emit()
	
	@QtCore.Slot()
	def on_delete(self):
		
		self.delete.emit()
	
	def on_data_source_changed(self, *args):
		
		self.update()
	
	def on_data_changed(self, *args):
		
		self.update()
