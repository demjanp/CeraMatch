
from PySide6 import (QtWidgets, QtCore, QtGui)

class Button(QtWidgets.QPushButton):
	
	def __init__(self, caption, callback):
		
		QtWidgets.QPushButton.__init__(self, caption)
		
		self.clicked.connect(callback)
