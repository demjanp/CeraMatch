from ceramatch.controller.cgraph_nodes.node import Node

from ceramatch.utils.fnc_drawing import (render_drawing, get_profile_data)

from deposit.utils.fnc_serialize import (try_numeric)
from deposit import DGeometry

from PySide6 import (QtWidgets, QtCore, QtGui, QtSvg)
import numbers

class SampleNode(Node):
	
	def __init__(self, cgraph, node_id, label, drawing_data, picture = None):
		
		Node.__init__(self, cgraph, node_id, label)
		
		self._drawing_data = drawing_data
		
		if picture is not None:
			self._picture = picture
		else:
			self._picture = QtGui.QPicture()
			painter = QtGui.QPainter(self._picture)
			render_drawing(
				self._drawing_data, painter, self.cgraph.LINE_WIDTH, 
				scale = self.cgraph.SCALE_DRAWINGS, color = QtCore.Qt.black,
			)
			painter.end()
		
		self._selection_polygon = QtGui.QPolygonF(
			QtCore.QRectF(self._picture.boundingRect().marginsAdded(
				QtCore.QMargins(3, 3, 3, 3)
			))
		)
		self._selection_shape = QtGui.QPainterPath()
		self._selection_shape.addPolygon(self._selection_polygon)
	
	def copy(self):
		
		return SampleNode(
			self.cgraph, self.node_id, self.label, 
			self._drawing_data, self._picture,
		)
	
	def get_drawing_data(self):
		
		return self._drawing_data.copy()
	
	def set_drawing_data(self, data):
		
		self._drawing_data = data
	
	def get_profile_data(self):
		# returns (coords, radius)
		
		coords, radius = get_profile_data(self._drawing_data)
		
		return (coords, radius)
	
	def boundingRect(self):
		
		scale = self.cgraph.get_scale_factor()
		if scale < self.cgraph.SCALE_CUTOFF:
			return QtCore.QRectF()
		
		return self._selection_polygon.boundingRect()
	
	def center(self):
		
		return self.boundingRect().center()
	
	def shape(self):
		
		return self._selection_shape
	
	def paint(self, painter, option, widget):
		
		scale = self.cgraph.get_scale_factor()
		if scale < self.cgraph.SCALE_CUTOFF:
			return
		
		selected = False
		color = QtCore.Qt.black
		if option.state & QtWidgets.QStyle.State_Selected:
			color = QtCore.Qt.red
			selected = True
		
		rect = self.boundingRect()
		bgcolor = QtGui.QColor("white")
		bgcolor.setAlphaF(0.8)
		painter.setBrush(QtGui.QBrush(bgcolor))
		painter.setPen(QtGui.QPen(bgcolor, 1))
		painter.drawRect(rect)
		
		painter.setBrush(QtGui.QBrush(color))
		painter.setPen(QtGui.QPen(color, 1))
		painter.drawPicture(0, 0, self._picture)
		
		if selected:
			painter.setBrush(QtGui.QBrush())
			painter.drawRect(rect)
	
	def update_tooltip(self):
		
		buffer = QtCore.QBuffer()
		buffer.open(QtCore.QIODevice.WriteOnly)
		scale = self.cgraph.SCALE_TOOLTIP / self.cgraph.SCALE_DRAWINGS
		gen = QtSvg.QSvgGenerator()
		gen.setOutputDevice(buffer)
		painter = QtGui.QPainter(gen)
		painter.scale(scale, scale)
		rect = self.boundingRect().marginsAdded(QtCore.QMargins(10, 10, 10, 10))
		rect = QtCore.QRectF(rect.x(), rect.y(), rect.width()*0.8, rect.height())
		painter.setPen(QtGui.QPen(QtCore.Qt.white, 0))
		painter.setBrush(QtGui.QBrush(QtCore.Qt.white))
		painter.drawRect(rect)
		painter.setPen(QtGui.QPen(QtCore.Qt.black, 0.5))
		painter.drawPicture(0, 0, self._picture)
		painter.drawText(rect, QtCore.Qt.AlignCenter | QtCore.Qt.AlignBottom, self.label)
		painter.end()
		self._tool_tip = "<img src=\"data:image/png;base64,%s\">" % (bytes(buffer.data().toBase64()).decode())
	
	def on_hover(self, state):
		# state: True = Enter, False = Leave
		
		if self._tool_tip is None:
			self.update_tooltip()
		
		self.cgraph.on_hover(self, state)

