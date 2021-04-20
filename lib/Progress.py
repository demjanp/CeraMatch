
from PySide2 import (QtWidgets, QtCore, QtGui)

class Progress(QtWidgets.QProgressDialog):
	
	def __init__(self, view):
		
		QtWidgets.QProgressDialog.__init__(self, "", "Cancel", 0, 0, view, flags = QtCore.Qt.FramelessWindowHint)
		
		self.setWindowModality(QtCore.Qt.WindowModal)
		
		self.cancel()
		self.reset()
	
	def cancel_pressed(self):
		
		return self.wasCanceled()
	
	def show(self, text = ""):
		
		self.setLabelText(text)
		
		QtWidgets.QProgressDialog.show(self)
		
		QtWidgets.QApplication.processEvents()
	
	def update_state(self, text = None, value = None, maximum = None):
		
		if not self.isVisible():
			self.show()
		if text is not None:
			self.setLabelText(text)
		if value is not None:
			self.setValue(value)
		if maximum is not None:
			self.setMaximum(maximum)
		
		QtWidgets.QApplication.processEvents()