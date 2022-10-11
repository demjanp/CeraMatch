
from PySide2 import (QtWidgets, QtCore, QtGui)

class Combo(QtWidgets.QComboBox):
	
	def __init__(self, callback, editable = False):
		
		QtWidgets.QComboBox.__init__(self)
		
		self.currentTextChanged.connect(callback)
		self.setEditable(editable)
	
	def clear_values(self):
		
		self.blockSignals(True)
		self.clear()
		self.blockSignals(False)
	
	def set_values(self, values, default = None):
		
		values = [str(val) for val in values]
		if (default is not None) and (default not in values):
			values = [default] + values
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
