
from ceramatch.view.vcontrols_groups.controls.button import (Button)

from PySide2 import (QtWidgets, QtCore, QtGui)

class Distances(QtWidgets.QGroupBox):
	
	signal_calculate = QtCore.Signal()
	signal_delete = QtCore.Signal()
	
	def __init__(self):
		
		QtWidgets.QGroupBox.__init__(self, "Distance")
		
		self.setStyleSheet("QGroupBox {font-weight: bold;}")
		self.setLayout(QtWidgets.QVBoxLayout())
		
		self.calculate_button = Button("Calculate Distances", self.on_calculate)
		self.delete_button = Button("Delete Distances", self.on_delete)
		
		self.layout().addWidget(self.calculate_button)
		self.layout().addWidget(self.delete_button)
	
	
	# ---- Signal handling
	# ------------------------------------------------------------------------
	@QtCore.Slot()
	def on_calculate(self):
		
		self.signal_calculate.emit()
	
	@QtCore.Slot()
	def on_delete(self):
		
		self.signal_delete.emit()
	
	
	# ---- get/set
	# ------------------------------------------------------------------------
	def set_calculate_distances_enabled(self, state):
		
		self.calculate_button.setEnabled(state)
	
	def set_delete_distances_enabled(self, state):
		
		self.delete_button.setEnabled(state)

