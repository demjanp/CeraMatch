from ceramatch.controller.cgraph_nodes.node import Node

from PySide6 import (QtWidgets, QtCore, QtGui, QtSvg)

class ClusterNode(Node):
	
	def __init__(self, cgraph, node_id, label, children):
		
		Node.__init__(self, cgraph, node_id, label)
		
		self._label_rect = QtGui.QFontMetrics(self.font).boundingRect(self.label)
		self._label_rect.moveLeft(-self._label_rect.center().x())
		self._label_rect.moveTop(self.radius)
	
	def set_label(self, label):
		
		self.label = str(label)
		self._label_rect = QtGui.QFontMetrics(self.font).boundingRect(self.label)
		self._label_rect.moveLeft(-self._label_rect.center().x())
		self._label_rect.moveTop(self.radius)
		self.update(self.boundingRect())
	
	def paint(self, painter, option, widget):
		
		scale = self.cgraph.get_scale_factor()
		color = QtCore.Qt.lightGray
		if (option.state & QtWidgets.QStyle.State_Selected):
			color = QtCore.Qt.red
		elif (option.state & QtWidgets.QStyle.State_HasFocus):
			color = QtCore.Qt.darkGray
		pen_width = 0
		if option.state & QtWidgets.QStyle.State_Sunken:
			pen_width = 2
		painter.setBrush(QtGui.QBrush(color))
		painter.setPen(QtGui.QPen(QtCore.Qt.darkGray, pen_width))
		radius = self.radius / scale
		painter.drawEllipse(-radius, -radius, 2*radius, 2*radius)
		if scale > 0.56:
			painter.setFont(self.font)
			painter.drawText(
				self._label_rect, 
				QtCore.Qt.AlignCenter | QtCore.Qt.AlignBottom, 
				self.label,
			)
	
	def boundingRect(self):
		
		scale = self.cgraph.get_scale_factor()
		radius = self.radius / scale
		return QtCore.QRectF(-radius, -radius, 2*radius, 2*radius).united(self._label_rect)
	
	def update_tooltip(self, gap = 5):
		
		nodes = list(self.get_children())
		n_side = len(nodes)
		if n_side > 3:
			n_side = max(1, int(n_side**0.5))
		x = 0
		y = 0
		h_max = 0
		n = 0
		w = 0
		x_max, y_max = 0, 0
		positions = []
		for node in nodes:
			positions.append([x + gap, y + gap])
			rect = node.boundingRect()
			w, h = rect.width() + gap, rect.height() + gap
			h_max = max(h_max, h)
			x += w
			n += 1
			if n > n_side:
				x = 0
				y += h_max
				h_max = 0
				n = 0
			x_max = max(x_max, x + w)
			y_max = max(y_max, y + h)
		if n_side == len(nodes):
			x_max -= w
		
		scale = self.cgraph.SCALE_TOOLTIP / self.cgraph.SCALE_DRAWINGS
		buffer = QtCore.QBuffer()
		buffer.open(QtCore.QIODevice.WriteOnly)
		gen = QtSvg.QSvgGenerator()
		gen.setOutputDevice(buffer)
		painter = QtGui.QPainter(gen)
		painter.scale(scale, scale)
		painter.setPen(QtGui.QPen(QtCore.Qt.white, 0))
		painter.setBrush(QtGui.QBrush(QtCore.Qt.white))
		rect = QtCore.QRectF(0, 0, x_max, y_max)
		painter.drawRect(rect)
		painter.setPen(QtGui.QPen(QtCore.Qt.black, 0.5))
		for i in range(len(nodes)):
			x, y = positions[i]
			painter.drawPicture(x, y, nodes[i]._picture)
		
		painter.drawText(rect, QtCore.Qt.AlignCenter | QtCore.Qt.AlignBottom, self.label)
		painter.end()
		self._tool_tip = "<img src=\"data:image/png;base64,%s\">" % (bytes(buffer.data().toBase64()).decode())
	
	def on_hover(self, state):
		# state: True = Enter, False = Leave
		
		if self._tool_tip is None:
			self.update_tooltip()
		
		self.cgraph.on_hover(self, state)
