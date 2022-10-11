
from PySide2 import (QtWidgets, QtCore, QtGui)

class LineEdit(QtWidgets.QLineEdit):
	
	def __init__(self, default = ""):
		
		QtWidgets.QLineEdit.__init__(self, default)
		
		self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
