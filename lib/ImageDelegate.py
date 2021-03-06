from PySide2 import (QtWidgets, QtCore, QtGui, QtSvg)

class ImageDelegate(QtWidgets.QStyledItemDelegate):
	
	def __init__(self, parent):
		
		self.parent = parent
		self.highlighted = False
		
		QtWidgets.QStyledItemDelegate.__init__(self, parent)
	
	def paint(self, painter, option, index):
		
		if self.highlighted:
			print("highlighted")
			self.highlighted = False
		
		sample = index.data(QtCore.Qt.UserRole)
		
		index2 = sample.index
		
		self.parent.list_model.on_paint(index2)
		
		QtWidgets.QStyledItemDelegate.paint(self, painter, option, index)

