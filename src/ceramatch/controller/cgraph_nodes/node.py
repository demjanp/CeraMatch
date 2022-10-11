from deposit_gui.dgui.dgraph_view import AbstractNode

from PySide2 import (QtWidgets, QtCore, QtGui)

class Node(AbstractNode):
	
	def __init__(
		self, cgraph, node_id, label = "", 
		radius = 10, font_family = "Calibri", font_size = 16,
	):
		
		AbstractNode.__init__(self, node_id, label, font_family, font_size)
		
		self.cgraph = cgraph
		self.radius = radius
		self._tool_tip = None
		self._moved = False
	
	def boundingRect(self):
		
		adjust = 2		
		return QtCore.QRectF(
			-self.radius - adjust, 
			-self.radius - adjust, 
			(2*self.radius) + 3 + adjust, 
			(2*self.radius) + 3 + adjust,
		)
	
	def center(self):
		
		return QtCore.QPointF(0, 0)
	
	def shape(self):
		
		path = QtGui.QPainterPath()
		path.addEllipse(-self.radius, -self.radius, 2*self.radius, 2*self.radius)
		return path
	
	def get_tool_tip(self):
		
		return self._tool_tip
	
	def paint(self, painter, option, widget):
		
		pen_width = 0
		if option.state & QtWidgets.QStyle.State_Sunken:
			pen_width = 2
		
		
		color = QtCore.Qt.white
		if (option.state & QtWidgets.QStyle.State_Selected):
			color = QtCore.Qt.red
		elif (option.state & QtWidgets.QStyle.State_HasFocus):
			color = QtCore.Qt.darkGray
		painter.setBrush(QtGui.QBrush(color))
		
		painter.setPen(QtGui.QPen(QtCore.Qt.black, pen_width))
		painter.drawEllipse(
			-self.radius, -self.radius, 2*self.radius, 2*self.radius
		)
	
	def on_hover(self, state):
		# state: True = Enter, False = Leave
		
		self.cgraph.on_hover(self, state)
	
	def on_position_change(self):
		
		self._moved = True
		for node in self.collidingItems():
			if not isinstance(node, Node):
				continue
			node.setFocus()
	
	def on_mouse_press(self):
		# re-implement
		
		pass
	
	def on_mouse_release(self):
		
		if self._moved:
			self._moved = False
			self.cgraph.on_moved(self)
		
		self.cgraph.on_mouse_released(self)
		
		for node in self.collidingItems():
			if not isinstance(node, Node):
				continue
			node.on_drop(self)
	
	def on_drop(self, node):
		
		self.cgraph.on_drop(node, self)

