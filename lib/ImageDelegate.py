from PySide2 import (QtWidgets, QtCore, QtGui, QtSvg)

class ImageDelegate(QtWidgets.QStyledItemDelegate):
	
	def __init__(self, parent):
		
		self.parent = parent
		
		QtWidgets.QStyledItemDelegate.__init__(self, parent)
	
	def paint(self, painter, option, index):
		
		sample = index.data(QtCore.Qt.UserRole)
		
		index2 = sample.index
		
		self.parent.list_model.on_paint(index2)
		
		QtWidgets.QStyledItemDelegate.paint(self, painter, option, index)
		
		icon = None
		if sample.outlier:
			icon = "res\\reject.svg"
		elif sample.central:
			icon = "res\\accept.svg"
		if icon is not None:
			renderer = QtSvg.QSvgRenderer(icon)
			renderer.render(painter, QtCore.QRectF(option.rect.x() + 10, option.rect.y() + 10, 36, 36))

