from PySide2 import (QtWidgets, QtCore, QtGui)

class StatusBar(QtWidgets.QStatusBar):

	def __init__(self, view):

		self.view = view
		self.model = view.model
		
		QtWidgets.QStatusBar.__init__(self, view)
		
	def message(self, text):

		self.showMessage(text)

