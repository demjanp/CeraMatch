
from PySide2 import (QtWidgets, QtCore, QtGui)

class Button(QtWidgets.QPushButton):
	
	def __init__(self, caption, callback, icon = None):
		
		QtWidgets.QPushButton.__init__(self, caption)
		
		if icon is not None:
			self.setIcon(QtGui.QIcon("res\%s" % (icon)))
		
		self.clicked.connect(callback)
