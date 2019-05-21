
from PySide2 import (QtWidgets, QtCore, QtGui)

class Button(QtWidgets.QPushButton):
	
	def __init__(self, caption, callback):
		
		QtWidgets.QPushButton.__init__(self, caption)
		
		self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
		self.setFixedWidth(200)
		self.clicked.connect(callback)
