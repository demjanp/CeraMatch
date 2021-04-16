from lib.fnc_drawing import *

from deposit.DModule import (DModule)
from deposit import Broadcasts

from PySide2 import (QtWidgets, QtCore, QtGui, QtPrintSupport)
from networkx.drawing.nx_agraph import graphviz_layout
from copy import copy
import networkx as nx
import numpy as np
import weakref

NODE_TYPE = QtWidgets.QGraphicsItem.UserType + 1
EDGE_TYPE = QtWidgets.QGraphicsItem.UserType + 2

SCALE_DRAWINGS = 0.5
LINE_WIDTH = 1

class Node(QtWidgets.QGraphicsItem):
	
	def __init__(self, node_id, radius = 10):
		
		self.node_id = node_id
		self.radius = radius
		self.edges = set([])  # [Edge, ...]
		self.cluster = None
		
		self._focused_items = set([])
		
		QtWidgets.QGraphicsItem.__init__(self)
		
		self.setAcceptHoverEvents(True)
		self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable)
		self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges)
		self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
		self.setFlag(QtWidgets.QGraphicsItem.ItemIsFocusable, False)
		self.setCacheMode(self.DeviceCoordinateCache)
		self.setZValue(-1)
	
	def type(self):
		
		return NODE_TYPE
	
	def add_edge(self, edge):
		
		self.edges.add(weakref.ref(edge))
		edge.adjust()
	
	def remove_edge(self, edge):
		
		edge = weakref.ref(edge)
		if edge in self.edges:
			self.edges.remove(edge)
	
	def set_cluster(self, node):
		
		if node is not None:
			node = weakref.ref(node)
		self.cluster = node
	
	def boundingRect(self):
		
		adjust = 2		
		return QtCore.QRectF(-self.radius - adjust, -self.radius - adjust, (2*self.radius) + 3 + adjust, (2*self.radius) + 3 + adjust)
	
	def center(self):
		
		return QtCore.QPointF(0, 0)
	
	def shape(self):
		# for collision detection
		
		path = QtGui.QPainterPath()
		path.addEllipse(-self.radius, -self.radius, 2*self.radius, 2*self.radius)
		return path
	
	def paint(self, painter, option, widget):
		
		pen_width = 0
		if option.state & QtWidgets.QStyle.State_Sunken:
			pen_width = 2
		if option.state & QtWidgets.QStyle.State_Selected:
			painter.setBrush(QtGui.QBrush(QtCore.Qt.lightGray))
		else:
			painter.setBrush(QtGui.QBrush(QtCore.Qt.white))
		painter.setPen(QtGui.QPen(QtCore.Qt.lightGray, pen_width))
		painter.drawEllipse(-self.radius, -self.radius, 2*self.radius, 2*self.radius)
	
	def hoverEnterEvent(self, event):
		
		self.scene().views()[0].on_hover_item(self)
	
	def itemChange(self, change, value):
		
		if change == QtWidgets.QGraphicsItem.ItemPositionChange:
			for edge in self.edges:
				edge().adjust()
			
			if (self.__class__ in [ClusterNode, SampleNode]):			
				for item in self._focused_items:
					item.clearFocus()
				self._focused_items.clear()
				for item in self.collidingItems():
					if isinstance(item, ClusterNode):
						if (self.cluster is None) or (item != self.cluster()):
							item.setFocus()
							self._focused_items.add(item)
		
		return QtWidgets.QGraphicsItem.itemChange(self, change, value)
	
	def mousePressEvent(self, event):
		
		self.update()
		QtWidgets.QGraphicsItem.mousePressEvent(self, event)
	
	def mouseReleaseEvent(self, event):
		
		def move_leaf(leaf, cluster):
			
			if leaf.cluster is not None:
				leaf.cluster().remove_child(leaf)
			
			y_mean = 0
			x_max = -np.inf
			for node in cluster.children:
				node = node()
				rect = node.boundingRect()
				pos = node.pos()
				x, y = pos.x(), pos.y()
				w = rect.width()
				y_mean += y
				x_max = max(x_max, x + w)
			y_mean /= len(cluster.children)
			leaf.setPos(x_max + 2, y_mean)
			
			cluster.add_child(leaf)
			leaf.set_cluster(cluster)
			view = self.scene().views()[0]
			view.remove_edges(leaf)
			view.add_edge(leaf, cluster)
			self.scene().clearSelection()
	
		self.update()
		QtWidgets.QGraphicsItem.mouseReleaseEvent(self, event)
		
		for item in self.collidingItems():
			if isinstance(item, ClusterNode):
				if (self.cluster is None) or (item != self.cluster()):
					if isinstance(self, ClusterNode):
						for leaf in copy(self.children):
							move_leaf(leaf(), item)
						self.scene().views()[0].remove_node(self)
					elif isinstance(self, SampleNode):
						cluster = self.cluster()
						move_leaf(self, item)
						if not cluster.children:
							self.scene().views()[0].remove_node(cluster)

class ClusterNode(Node):
	
	def __init__(self, node_id, radius = 15):
		
		self.children = set([])
		self._prev_pos = None
		
		Node.__init__(self, node_id, radius)
		
		self.setFlag(QtWidgets.QGraphicsItem.ItemIsFocusable, True)
	
	def set_children(self, children):
		
		self.children.clear()
		for child in children:
			child.set_cluster(self)
			self.children.add(weakref.ref(child))
	
	def add_child(self, node):
		
		node_ref = weakref.ref(node)
		if node_ref not in self.children:
			self.children.add(node_ref)
	
	def remove_child(self, node):
		
		node_ref = weakref.ref(node)
		if node_ref in self.children:
			self.children.remove(node_ref)
	
	def paint(self, painter, option, widget):
		
		pen_width = 0
		if option.state & QtWidgets.QStyle.State_Sunken:
			pen_width = 2
		
		if (option.state & QtWidgets.QStyle.State_Selected) or (option.state & QtWidgets.QStyle.State_HasFocus):
			painter.setBrush(QtGui.QBrush(QtCore.Qt.darkGray))
		else:
			painter.setBrush(QtGui.QBrush(QtCore.Qt.lightGray))
		painter.setPen(QtGui.QPen(QtCore.Qt.darkGray, pen_width))
		painter.drawEllipse(-self.radius, -self.radius, 2*self.radius, 2*self.radius)
	
	def mousePressEvent(self, event):
		
		Node.mousePressEvent(self, event)
		for node in self.children:
			node().setSelected(True)
		self.setSelected(True)
	
	def mouseReleaseEvent(self, event):
		
		Node.mouseReleaseEvent(self, event)
		
		for node in self.children:
			node().setSelected(False)
		self.setSelected(False)
	
	def mouseDoubleClickEvent(self, event):
		
		Node.mouseDoubleClickEvent(self, event)

class SampleNode(Node):
	
	def __init__(self, sample_id, descriptors):
		
		Node.__init__(self, sample_id)
		
		self.picture = QtGui.QPicture()
		painter = QtGui.QPainter(self.picture)
		render_drawing(descriptors, painter, LINE_WIDTH, scale = SCALE_DRAWINGS, color = QtCore.Qt.black)
		painter.end()
		
		self._rect = QtCore.QRectF(self.picture.boundingRect().marginsAdded(QtCore.QMargins(3, 3, 3, 3)))
		
		self.setFlag(QtWidgets.QGraphicsItem.ItemIsFocusable, True)
	
	def boundingRect(self):
		
		return self._rect
	
	def center(self):
		
		return QtCore.QPointF(self._rect.width() / 2, -5)
	
	def shape(self):
		
		path = QtGui.QPainterPath()
		path.addRect(self._rect)
		return path
	
	def paint(self, painter, option, widget):
		
		box_color = None
		if option.state & QtWidgets.QStyle.State_Sunken:
			box_color = QtCore.Qt.gray
		elif option.state & QtWidgets.QStyle.State_Selected:
			box_color = QtCore.Qt.red
		if box_color is not None:
			painter.setPen(QtGui.QPen(box_color, 1))
			painter.drawRect(self._rect)
		painter.drawPicture(0, 0, self.picture)

class Edge(QtWidgets.QGraphicsItem):
	
	def __init__(self, source, target):
		
		self.source = weakref.ref(source)
		self.target = weakref.ref(target)
		
		self.source_point = QtCore.QPointF()
		self.target_point = QtCore.QPointF()
		self.line = None
		
		QtWidgets.QGraphicsItem.__init__(self)
		
		self.setZValue(-2)
		
		self.source().add_edge(self)
		self.target().add_edge(self)
		self.adjust()
	
	def type(self):
		
		return EDGE_TYPE
	
	def boundingRect(self):
		
		if self.line is None:
			return QtCore.QRectF()
		x1, y1, x2, y2 = self.line.x1(), self.line.y1(), self.line.x2(), self.line.y2()
		return QtCore.QRectF(min(x1, x2), min(y1, y2), abs(x1 - x2), abs(y1 - y2))
	
	def adjust(self):
		
		if not self.source() or not self.target():
			return
		
		self.line = QtCore.QLineF(self.mapFromItem(self.source(), self.source().center()), self.mapFromItem(self.target(), self.target().center()))
		length = self.line.length()
		
		if length == 0:
			return
		
		self.prepareGeometryChange()
		self.source_point = self.line.p1()
		self.target_point = self.line.p2()
	
	def paint(self, painter, option, widget):
		
		if not self.source() or not self.target():
			return
		
		painter.setPen(QtGui.QPen(QtCore.Qt.lightGray, LINE_WIDTH, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
		painter.drawLine(self.line)

class GraphView(DModule, QtWidgets.QGraphicsView):
	
	activated = QtCore.Signal(object)
	dropped = QtCore.Signal(object, object)  # (source, target)
	
	def __init__(self, view):
		
		self.view = view
		self.model = view.model
		
		self._nodes = {}  # {node_id: Node, ...}
		self._edges = []  # [Edge, ...]
		self._labels = {} # {node_id: label, ...}
		self._mouse_prev = None
		
		DModule.__init__(self)
		QtWidgets.QGraphicsView.__init__(self)
		
		scene = QtWidgets.QGraphicsScene(self)
		scene.setItemIndexMethod(QtWidgets.QGraphicsScene.NoIndex)
		self.setScene(scene)
		self.setRenderHint(QtGui.QPainter.Antialiasing)
		self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
		self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
		self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
		
		self.setMinimumSize(400, 400)
		
		scene.selectionChanged.connect(self.on_selected)
		
		self.connect_broadcast(Broadcasts.STORE_LOADED, self.on_store_changed)
		self.connect_broadcast(Broadcasts.STORE_DATA_SOURCE_CHANGED, self.on_store_changed)
	
	def clear(self):
		
		self.scene().clear()
		self.setSceneRect(QtCore.QRectF())
		self._nodes.clear()
		self._edges.clear()
		self._labels.clear()
	
	def reset_scene(self):
		
		rect = self.scene().itemsBoundingRect().marginsAdded(QtCore.QMarginsF(10, 10, 10, 10))
		self.setSceneRect(rect)
		self.fitInView(rect, QtCore.Qt.KeepAspectRatio)
	
	def scale_view(self, factor):
		
		f = self.matrix().scale(factor, factor).mapRect(QtCore.QRectF(0, 0, 1, 1)).width()
		
		self.scale(factor, factor)
	
	def remove_node(self, node):
		
		node_id = node.node_id
		self.remove_edges(node)
		self.scene().removeItem(node)
		del self._nodes[node_id]
	
	def add_edge(self, source_node, target_node):
		
		self._edges.append(Edge(source_node, target_node))
		self.scene().addItem(self._edges[-1])
	
	def remove_edges(self, node):
		
		idxs = []
		node_ref = weakref.ref(node)
		for idx, edge in enumerate(self._edges):
			if (edge.source == node_ref) or (edge.target == node_ref):
				idxs.append(idx)
		idxs = sorted(idxs)[::-1]
		for idx in idxs:
			for node2 in [self._edges[idx].source(), self._edges[idx].target()]:
				node2.remove_edge(self._edges[idx])
			self.scene().removeItem(self._edges[idx])
			del self._edges[idx]
	
	def set_data(self, sample_data, nodes, edges, clusters = {}, labels = {}, positions = None, gap = 10):
		# sample_data = {sample_id: [obj_id, descriptors], ...}
		# nodes = [node_id, ...]
		# edges = [[source_id, target_id], ...]
		# clusters = {node_id: [sample_id, ...], ...}
		# labels = {node_id: label, ...}
		# positions = {node_id: (x, y), ...} or None
		
		def get_positions():
			
			positions = {}
			done = set([])
			y_max = 0
			G = nx.DiGraph()
			for source_id, target_id in edges:
				G.add_edge(source_id, target_id)
				done.add(source_id)
				done.add(target_id)
			for node_id in done:
				if node_id not in sample_data:
					continue
				rect = self._nodes[node_id].boundingRect()
				mul = 0.013  # TODO find a way to calculate
				w, h = rect.width(), rect.height()
				G.nodes[node_id]["width"] = (w + 2*gap) * mul
				G.nodes[node_id]["height"] = (h + 2*gap) * mul
				
			g_positions = graphviz_layout(G, prog = "graphviz/dot.exe")
			xmin, ymax = np.inf, -np.inf
			for node_id in g_positions:
				x, y = g_positions[node_id]
				xmin = min(xmin, x)
				ymax = max(ymax, y)
			for node_id in g_positions:
				x, y = g_positions[node_id]
				g_positions[node_id] = (x - xmin, ymax - y)
			
			for node_id in g_positions:
				positions[node_id] = tuple(g_positions[node_id])
				y_max = max(y_max, positions[node_id][1])
			
			n = len([node_id for node_id in self._nodes if node_id not in done])
			if n > 3:
				n_side = max(1, int(n**0.5))
			else:
				n_side = n
			
			x = 0
			y = y_max
			h_max = 0
			n = 0
			for node_id in self._nodes:
				if node_id in done:
					continue
				positions[node_id] = (x, y)
				rect = self._nodes[node_id].boundingRect()
				w, h = rect.width() + gap, rect.height() + gap
				h_max = max(h_max, h)
				x += w
				n += 1
				if n > n_side:
					x = 0
					y += h_max
					h_max = 0
					n = 0
			
			return positions
		
		self.clear()
		
		self._labels = labels
		nodes = set(nodes)
		nodes.update(list(sample_data.keys()))
		for node_id in nodes:
			if node_id in sample_data:
				self._nodes[node_id] = SampleNode(node_id, sample_data[node_id][1])
			elif node_id in clusters:
				self._nodes[node_id] = ClusterNode(node_id)
			else:
				self._nodes[node_id] = Node(node_id)
			self.scene().addItem(self._nodes[node_id])
		
		for node_id in self._nodes:
			if isinstance(self._nodes[node_id], ClusterNode):
				self._nodes[node_id].set_children([self._nodes[node] for node in clusters[node_id]])
		
		if (not positions) or (False in [(node_id in positions) for node_id in nodes]):
			positions = get_positions()
		for node_id in self._nodes:
			self._nodes[node_id].setPos(*positions[node_id])
		
		for source_id, target_id in edges:
			self._edges.append(Edge(self._nodes[source_id], self._nodes[target_id]))
			self.scene().addItem(self._edges[-1])
		
		self.reset_scene()
	
	def on_hover_item(self, item):
		
		if item.node_id in self._labels:
			self.view.statusbar.message("%s - %s" % (self._labels[item.node_id], item.node_id))
		else:
			self.view.statusbar.message(str(item.node_id))
	
	@QtCore.Slot()
	def on_selected(self):
		
		pass
	
	def on_store_changed(self, *args):
		
		self.clear()
	
	def wheelEvent(self, event):
		
		self.scale_view(2**(event.delta() / 240.0))
	
	def mousePressEvent(self, event):
		
		if event.button() == QtCore.Qt.LeftButton:
			self.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
			self.setCursor(QtCore.Qt.ArrowCursor)
		elif event.button() == QtCore.Qt.RightButton:
			self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
			self.setCursor(QtCore.Qt.OpenHandCursor)
			self._mouse_prev = (event.x(), event.y())
		
		QtWidgets.QGraphicsView.mousePressEvent(self, event)
	
	def mouseMoveEvent(self, event):
		
		if self._mouse_prev is not None:
			prev_point = self.mapToScene(*self._mouse_prev)
			new_point = self.mapToScene(event.pos())
			translation = new_point - prev_point
			self.translate(translation.x(), translation.y())
			self._mouse_prev = (event.x(), event.y())
		
		QtWidgets.QGraphicsView.mouseMoveEvent(self, event)
	
	def mouseReleaseEvent(self, event):
		
		self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
		self.setCursor(QtCore.Qt.ArrowCursor)
		self._mouse_prev = None
		
		QtWidgets.QGraphicsView.mouseReleaseEvent(self, event)
	
