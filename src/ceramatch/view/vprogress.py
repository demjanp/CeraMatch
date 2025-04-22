from PySide6 import (QtWidgets, QtCore, QtGui)

class VProgress(QtCore.QObject):
	
	def __init__(self, view):
		
		self._view = view
		self._text = ""
		self._maximum = 0
		self._cancel_pressed = False
		
		QtCore.QObject.__init__(self)
	
	def reset(self):
		
		self._text = ""
		self._maximum = 0
		self._cancel_pressed = False
	
	def cancel_pressed(self):
		
		return self._cancel_pressed
	
	def show(self, text = ""):
		
		self.reset()
		QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
	
	def stop(self):
		
		self.reset()
		QtWidgets.QApplication.restoreOverrideCursor()
	
	def update_state(self, text = None, value = None, maximum = None):
		
		if text is not None:
			self._text = text
		if value is None:
			value = 0
		if maximum is not None:
			self._maximum = maximum
		
		print("\r%s %s/%s           " % (self._text, value, self._maximum), end = '')
		
		if value >= self._maximum:
			self.stop()

