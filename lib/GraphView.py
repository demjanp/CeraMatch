
from lib.fnc_drawing import *

from lib.History import (History)

from deposit.DModule import (DModule)
from deposit import Broadcasts

from PySide2 import (QtWidgets, QtCore, QtGui, QtSvg, QtPrintSupport)
from networkx.drawing.nx_agraph import graphviz_layout
from copy import copy
import networkx as nx
import numpy as np
import weakref

NODE_TYPE = QtWidgets.QGraphicsItem.UserType + 1
EDGE_TYPE = QtWidgets.QGraphicsItem.UserType + 2

SCALE_DRAWINGS = 0.5
SCALE_TOOLTIP = 1
SCALE_CUTOFF = 0.2
LINE_WIDTH = 1
TOOLTIP_SHOW_DELAY = 800
TOOLTIP_HIDE_DELAY = 400

class ToolTip(QtWidgets.QLabel):
	
	def __init__(self):
		
		self.node = None
		
		QtWidgets.QLabel.__init__(self)
		
		self.setWindowFlags(QtCore.Qt.ToolTip)
		self.setFrameShape(QtWidgets.QFrame.StyledPanel)
		self.setStyleSheet("QLabel { background-color : white; }")
		
		self.hide_timer = QtCore.QTimer(singleShot = True, timeout = self.on_hide_timer)
		self.show_timer = QtCore.QTimer(singleShot = True, timeout = self.on_show_timer)
		self.hide()
	
	def move(self, pos):
		
		self.ensurePolished()
		geo = QtCore.QRect(pos, self.sizeHint())
		try:
			screen = QtWidgets.QApplication.screenAt(pos)
		except:
			for screen in QtWidgets.QApplication.screens():
				if pos in screen.geometry():
					break
			else:
				screen = None
		if not screen:
			screen = QtWidgets.QApplication.primaryScreen()
		screenGeo = screen.availableGeometry()
		if geo.bottom() > screenGeo.bottom():
			geo.moveBottom(screenGeo.bottom())
		if geo.top() < screenGeo.top():
			geo.moveTop(screenGeo.top())
		if geo.right() > screenGeo.right():
			geo.moveRight(screenGeo.right())
		if geo.left() < screenGeo.left():
			geo.moveLeft(screenGeo.left())
		QtWidgets.QLabel.move(self, geo.topLeft())
	
	def show(self, text, node):
		
		self.show_timer.stop()
		self.hide_timer.stop()
		self.node = weakref.ref(node)
		self.setText(text)
		self.adjustSize()
		self.move(QtGui.QCursor.pos())
		self.show_timer.start(TOOLTIP_SHOW_DELAY)
	
	def hide(self, delay = None):
		
		self.show_timer.stop()
		if delay is None:
			self.hide_timer.stop()
			QtWidgets.QLabel.hide(self)
			return
		if self.underMouse():
			return
		self.hide_timer.start(delay)
	
	@QtCore.Slot()
	def on_show_timer(self):
		
		if self.node().isUnderMouse():
			QtWidgets.QLabel.show(self)
	
	@QtCore.Slot()
	def on_hide_timer(self):
		
		self.hide()
	
	def leaveEvent(self, event):
		
		if self.isVisible():
			self.hide(TOOLTIP_HIDE_DELAY)
		else:
			self.show_timer.stop()
		
		QtWidgets.QLabel.leaveEvent(self, event)
	
	def mousePressEvent(self, event):
		
		self.hide()
		
		QtWidgets.QLabel.mousePressEvent(self, event)

class Node(QtWidgets.QGraphicsItem):
	
	def __init__(self, model, graph_view, node_id, label, radius = 10):
		
		self.model = weakref.ref(model)
		self.graph_view = graph_view
		
		self.node_id = node_id
		self.label = label
		self.radius = radius
		self.edges = set([])  # [Edge, ...]
		self.cluster = None
		
		self._focused_items = set([])
		self.moved = False
		self.tool_tip = None
		
		QtWidgets.QGraphicsItem.__init__(self)
		
		self.setAcceptHoverEvents(True)
		self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable)
		self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges)
		self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, False)
		self.setFlag(QtWidgets.QGraphicsItem.ItemIsFocusable, False)
		self.setCacheMode(self.DeviceCoordinateCache)
		self.setZValue(-1)
	
	def store_position(self):
		
		pos = self.pos()
		self.model().clustering.store_position(self.node_id, pos.x(), pos.y())
	
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
	
	def set_pos(self, x, y):
		
		self.setPos(x, y)
		self.store_position()
	
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
		
		self.graph_view.on_hover_item(self)
		
		QtWidgets.QGraphicsItem.hoverEnterEvent(self, event)
	
	def hoverLeaveEvent(self, event):
		
		self.graph_view.on_hover_leave()
		
		QtWidgets.QGraphicsItem.hoverLeaveEvent(self, event)
	
	def itemChange(self, change, value):
		
		if change == QtWidgets.QGraphicsItem.ItemPositionChange:
			
			self.moved = True
			
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
			leaf.set_pos(x_max + 2, y_mean)
			
			cluster.add_child(leaf)
			leaf.set_cluster(cluster)
			self.graph_view.remove_edges(leaf)
			self.graph_view.add_edge(leaf, cluster)
			self.scene().clearSelection()
	
		self.update()
		QtWidgets.QGraphicsItem.mouseReleaseEvent(self, event)
		
		if self.moved:
			self.graph_view.store_positions()
		
		changed = False
		for item in self.collidingItems():
			if isinstance(item, ClusterNode):
				if (self.cluster is None) or (item != self.cluster()):
					if isinstance(self, ClusterNode):
						for leaf in copy(self.children):
							move_leaf(leaf(), item)
						self.graph_view.remove_node(self)
						changed = True
					
					elif isinstance(self, SampleNode):
						cluster = self.cluster()
						move_leaf(self, item)
						if not cluster.children:
							self.graph_view.remove_node(cluster)
						changed = True
		self.graph_view.history.save()
		self.graph_view.broadcast(Broadcasts.VIEW_ACTION)

class ClusterNode(Node):
	
	def __init__(self, model, graph_view, node_id, label, radius = 5):
		
		self.children = set([])  # {weakref.ref(node), ...}
		self._first_sample = None
		
		Node.__init__(self, model, graph_view, node_id, label, radius)
		
		self.setFlag(QtWidgets.QGraphicsItem.ItemIsFocusable, True)
	
	def shape(self):
		# for collision detection
		
		path = QtGui.QPainterPath()
		scale = self.graph_view.get_scale()
		radius = self.radius / scale
		path.addEllipse(-radius, -radius, 2*radius, 2*radius)
		return path
	
	def boundingRect(self):
		
		adjust = 2		
		scale = self.graph_view.get_scale()
		radius = self.radius / scale
		return QtCore.QRectF(-radius - adjust, -radius - adjust, (2*radius) + 3 + adjust, (2*radius) + 3 + adjust)
	
	def set_children(self, children):
		
		self.children.clear()
		self.tool_tip = None
		self.update_first_sample()
		for child in children:
			child.set_cluster(self)
			self.children.add(weakref.ref(child))
	
	def add_child(self, node):
		
		self.tool_tip = None
		self.update_first_sample()
		node_ref = weakref.ref(node)
		if node_ref not in self.children:
			self.children.add(node_ref)
			self.model().clustering.add_child(self.node_id, node.node_id)
	
	def remove_child(self, node):
		
		self.tool_tip = None
		self.update_first_sample()
		node_ref = weakref.ref(node)
		if node_ref in self.children:
			self.children.remove(node_ref)
			self.model().clustering.remove_child(self.node_id, node.node_id)
	
	def paint(self, painter, option, widget):
		
		scale = self.graph_view.get_scale()
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
	
	def update_first_sample(self):
		
		self._first_sample = self.model().clustering.get_first_sample(self.node_id)
	
	def update_tooltip(self, gap = 5):
		
		nodes = [node() for node in self.children]
		n_side = len(nodes)
		if n_side > 3:
			n_side = max(1, int(n_side**0.5))
		x = 0
		y = 0
		h_max = 0
		n = 0
		x_max, y_max = 0, 0
		positions = []
		for node in nodes:
			positions.append([x + gap, y + gap])
			w, h = node._rect.width()*0.8 + gap, node._rect.height() + gap
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
		
		scale = SCALE_TOOLTIP / SCALE_DRAWINGS
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
			painter.drawPicture(x, y, nodes[i].picture)
		
		painter.drawText(rect, QtCore.Qt.AlignCenter | QtCore.Qt.AlignBottom, self.label)
		painter.end()
		self.tool_tip = "<img src=\"data:image/png;base64,%s\">" % (bytes(buffer.data().toBase64()).decode())
	
	def hoverEnterEvent(self, event):
		
		if (self.tool_tip is None) and self.children:
			self.update_tooltip()
		
		Node.hoverEnterEvent(self, event)
	
	def mousePressEvent(self, event):
		
		Node.mousePressEvent(self, event)
		
		self.graph_view.scene().clearSelection()
		for node in self.children:
			node().setSelected(True)
		self.setSelected(True)
	
	def mouseReleaseEvent(self, event):
		
		Node.mouseReleaseEvent(self, event)
		
		for node in self.children:
			node().setSelected(False)
		self.setSelected(False)
	
class SampleNode(Node):
	
	def __init__(self, model, graph_view, sample_id, label, descriptors):
		
		Node.__init__(self, model, graph_view, sample_id, label)
		
		self.picture = QtGui.QPicture()
		painter = QtGui.QPainter(self.picture)
		render_drawing(descriptors, painter, LINE_WIDTH, scale = SCALE_DRAWINGS, color = QtCore.Qt.black)
		painter.end()
		
		self._rect = QtCore.QRectF(self.picture.boundingRect().marginsAdded(QtCore.QMargins(3, 3, 3, 3)))
		
		self.setFlag(QtWidgets.QGraphicsItem.ItemIsFocusable, True)
		self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
	
	def boundingRect(self):
		
		return self._rect
	
	def center(self):
		
		return QtCore.QPointF(self._rect.width() / 2, -5)
	
	def shape(self):
		
		path = QtGui.QPainterPath()
		path.addRect(self._rect)
		return path
	
	def paint(self, painter, option, widget):
		
		selected = False
		color = QtCore.Qt.black
		if option.state & QtWidgets.QStyle.State_Selected:
			color = QtCore.Qt.red
			selected = True
		
		scale = self.graph_view.get_scale()
		if scale < SCALE_CUTOFF:
			painter.setBrush(QtGui.QBrush(color))
			radius = 1 / scale
			x = self._rect.width() / 2
			painter.drawEllipse(x - radius, -radius, 2*radius, 2*radius)
		else:
			bgcolor = QtGui.QColor("white")
			bgcolor.setAlphaF(0.8)
			painter.setBrush(QtGui.QBrush(bgcolor))
			painter.setPen(QtGui.QPen(bgcolor, 1))
			painter.drawRect(self._rect)
			
			painter.setBrush(QtGui.QBrush(color))
			painter.setPen(QtGui.QPen(color, 1))
			painter.drawPicture(0, 0, self.picture)
			
			if selected:
				painter.setBrush(QtGui.QBrush())
				painter.drawRect(self._rect)

	def update_tooltip(self):
		
		buffer = QtCore.QBuffer()
		buffer.open(QtCore.QIODevice.WriteOnly)
		scale = SCALE_TOOLTIP / SCALE_DRAWINGS
		gen = QtSvg.QSvgGenerator()
		gen.setOutputDevice(buffer)
		painter = QtGui.QPainter(gen)
		painter.scale(scale, scale)
		rect = self._rect.marginsAdded(QtCore.QMargins(10, 10, 10, 10))
		rect = QtCore.QRectF(rect.x(), rect.y(), rect.width()*0.8, rect.height())
		painter.setPen(QtGui.QPen(QtCore.Qt.white, 0))
		painter.setBrush(QtGui.QBrush(QtCore.Qt.white))
		painter.drawRect(rect)
		painter.setPen(QtGui.QPen(QtCore.Qt.black, 0.5))
		painter.drawPicture(0, 0, self.picture)
		painter.drawText(rect, QtCore.Qt.AlignCenter | QtCore.Qt.AlignBottom, self.label)
		painter.end()
		self.tool_tip = "<img src=\"data:image/png;base64,%s\">" % (bytes(buffer.data().toBase64()).decode())
	
	def hoverEnterEvent(self, event):
		
		if self.tool_tip is None:
			self.update_tooltip()
		
		Node.hoverEnterEvent(self, event)
	
	def mouseDoubleClickEvent(self, event):
		
		model = self.model()
		model.launch_deposit()
		obj_id = model.sample_data[self.node_id[1:]][0]
		model.dc.view.query("SELECT Sample.* WHERE id(Sample) == %d" % (obj_id))

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
		
		pen = QtGui.QPen(QtCore.Qt.lightGray, LINE_WIDTH, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
		pen.setCosmetic(True)
		painter.setPen(pen)
		painter.drawLine(self.line)

class GraphView(DModule, QtWidgets.QGraphicsView):
	
	activated = QtCore.Signal(object)
	dropped = QtCore.Signal(object, object)  # (source, target)
	
	def __init__(self, view):
		
		self.view = view
		self.model = view.model
		
		self.history = History(self.model.clustering)
		
		self._nodes = {}  # {node_id: Node, ...}
		self._edges = []  # [Edge, ...]
		self._labels = {} # {node_id: label, ...}
		self._mouse_prev = None
		self._tool_tip = ToolTip()
		
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
		self.history.clear()
	
	def reset_scene(self):
		
		rect = self.scene().itemsBoundingRect().marginsAdded(QtCore.QMarginsF(10, 10, 10, 10))
		self.setSceneRect(rect)
		self.fitInView(rect, QtCore.Qt.KeepAspectRatio)
	
	def scale_view(self, factor):
		
		self.scale(factor, factor)
	
	def get_scale(self):
		
		return self.transform().m11()
	
	def get_selected(self):
		
		return self.scene().selectedItems()
	
	def remove_node(self, node):
		
		node_id = node.node_id
		self.remove_edges(node)
		self.scene().removeItem(node)
		del self._nodes[node_id]
		self.model.clustering.remove(node_id)
	
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
	
	def store_positions(self):
		
		for node_id in self._nodes:
			if self._nodes[node_id].moved:
				self._nodes[node_id].store_position()
				self._nodes[node_id].moved = False
	
	def set_data(self, sample_data, clusters = {}, nodes = [], edges = [], labels = {}, positions = {}, gap = 10, x_stretch = 1, y_stretch = 2):
		# sample_data = {sample_id: [obj_id, descriptors], ...}
		# clusters = {node_id: [sample_id, ...], ...}
		# nodes = [node_id, ...]
		# edges = [[source_id, target_id], ...]
		# labels = {node_id: label, ...}
		# positions = {node_id: (x, y), ...} or None
		
		def calc_positions(edges):
			
			pos_collect = {}
			done = set([])
			y_max = 0
			G = nx.DiGraph()
			for source_id, target_id in edges:
				G.add_edge(source_id, target_id)
				done.add(source_id)
				done.add(target_id)
			for node_id in done:
				if not node_id.startswith("@"):
					continue
				if node_id[1:] not in sample_data:
					continue
				rect = self._nodes[node_id].boundingRect()
				mul = 0.013  # TODO find a way to calculate
				w, h = rect.width(), rect.height()
				G.nodes[node_id]["width"] = (w + 2*gap) * mul
				G.nodes[node_id]["height"] = (h + 2*gap) * mul
			
			g_positions = graphviz_layout(G, prog = "graphviz/dot.exe")
			for node_id in g_positions:
				x, y = g_positions[node_id]
				g_positions[node_id] = (x * x_stretch, y * y_stretch)
			xmin, ymax = np.inf, -np.inf
			for node_id in g_positions:
				x, y = g_positions[node_id]
				xmin = min(xmin, x)
				ymax = max(ymax, y)
			for node_id in g_positions:
				x, y = g_positions[node_id]
				g_positions[node_id] = (x - xmin, ymax - y)
			
			for node_id in g_positions:
				pos_collect[node_id] = tuple(g_positions[node_id])
				y_max = max(y_max, pos_collect[node_id][1])
			
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
				pos_collect[node_id] = (x, y)
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
			
			return pos_collect
		
		self.clear()
		
		if not nodes:
			nodes = set([])
			for sample_id in sample_data:
				node_id = "@%s" % (sample_id)
				nodes.add(node_id)
				labels[node_id] = str(sample_id)
		
		self._labels = labels
		nodes = set(nodes)
		nodes.update(["@%s" % (sample_id) for sample_id in sample_data])
		
		cmax = len(nodes)
		self.view.progress.update_state(text = "Drawing Tree...", value = 1, maximum = cmax)
		
		for node_id in nodes:
			if node_id.startswith("@") and (node_id[1:] in sample_data):
				self._nodes[node_id] = SampleNode(self.model, self, node_id, self._labels[node_id], sample_data[node_id[1:]][1])
			elif node_id in clusters:
				self._nodes[node_id] = ClusterNode(self.model, self, node_id, self._labels[node_id])
			else:
				self._nodes[node_id] = Node(self.model, self, node_id, self._labels[node_id])
		
		for node_id in self._nodes:
			if isinstance(self._nodes[node_id], ClusterNode):
				self._nodes[node_id].set_children([self._nodes[node] for node in clusters[node_id]])
		
		if False in [(node_id in positions) for node_id in nodes]:
			positions_new = calc_positions(edges)
			for node_id in positions:
				positions_new[node_id] = positions[node_id]
			positions = positions_new
		
		cnt = 1
		for node_id in self._nodes:
			self.view.progress.update_state(text = "Drawing Tree...", value = cnt, maximum = cmax)
			cnt += 1
			if self.view.progress.cancel_pressed():
				self.clear()
				return
			
			self._nodes[node_id].set_pos(*positions[node_id])
			self.scene().addItem(self._nodes[node_id])
		
		for source_id, target_id in edges:
			self._edges.append(Edge(self._nodes[source_id], self._nodes[target_id]))
			self.scene().addItem(self._edges[-1])
		
		self.model.clustering.set_data(clusters, nodes, edges, labels, positions)
		
		self.history.save()
		self.broadcast(Broadcasts.VIEW_ACTION)
	
	def add_cluster(self):
		
		children = copy(self.get_selected())
		
		x_mean = 0
		y_max = -np.inf
		w_max = 0
		h_max = 0
		for node in children:
			rect = node.boundingRect()
			pos = node.pos()
			x, y = pos.x(), pos.y()
			h = rect.height()
			w = rect.width()
			x_mean += x
			y_max = max(y_max, y + h)
			w_max = max(w_max, w)
			h_max = max(h_max, h)
		x_mean /= len(children)
		cx, cy = x_mean, y_max + 3
		cluster_id = "#%d" % (max([int(node_id[1:]) if node_id.startswith("#") else 0 for node_id in self._nodes]) + 1)
		self._labels[cluster_id] = "Manual_%s" % (cluster_id[1:])
		self._nodes[cluster_id] = ClusterNode(self.model, self, cluster_id, self._labels[cluster_id])
		self.scene().addItem(self._nodes[cluster_id])
		self._nodes[cluster_id].set_pos(cx, cy)
		
		cluster = self._nodes[cluster_id]
		dx = 0
		side = 1
		cy += h_max
		for node in children:
			dx += w_max / 2
			side *= -1
			node.set_pos(cx + dx * side, cy)
			old_cluster = node.cluster
			if old_cluster is not None:
				old_cluster = old_cluster()
				old_cluster.remove_child(node)
				if not old_cluster.children:
					self.remove_node(old_cluster)
					self.model.clustering.remove(old_cluster.node_id)
			cluster.add_child(node)
			node.set_cluster(cluster)
			self.remove_edges(node)
			self.add_edge(node, cluster)
		
		self.model.clustering.add_cluster(cluster_id, [node.node_id for node in children])
		
		self.history.save()
		self.broadcast(Broadcasts.VIEW_ACTION)
	
	def undo(self):
		
		if not self.history.can_undo():
			return
		self.set_data(self.model.sample_data, *self.history.undo())
	
	def redo(self):
		
		if not self.history.can_redo():
			return
		self.set_data(self.model.sample_data, *self.history.redo())
	
	def save_pdf(self, path, dpi = 200):
		
		cmax = 4
		self.view.progress.update_state(text = "Rendering...", value = 1, maximum = cmax)
		if self.view.progress.cancel_pressed():
			return
		
		self.scene().clearSelection()
		
		scale = self.get_scale()
		scale_tgt = SCALE_CUTOFF * 2
		if scale < scale_tgt:
			self.scale_view(scale_tgt / scale)
		
		rect = self.scene().itemsBoundingRect()
		m = min(rect.width(), rect.height())*0.05
		rect = rect.marginsAdded(QtCore.QMarginsF(m, m, m, m))
		w, h = rect.width(), rect.height()
		
		self.view.progress.update_state(text = "Rendering...", value = 2, maximum = cmax)
		if self.view.progress.cancel_pressed():
			return
		
		printer = QtPrintSupport.QPrinter()
		printer.setWinPageSize(QtGui.QPageSize.A4)
		printer.setFullPage(True)
		printer.setOrientation(QtPrintSupport.QPrinter.Landscape)
		printer.setOutputFormat(QtPrintSupport.QPrinter.PdfFormat)
		printer.setOutputFileName(path)
		size = printer.pageLayout().pageSize().sizePoints()
		pw, ph = size.height(), size.width()
		scale = min(pw / w, ph / h)
		printer.setResolution(int(round(dpi / scale)))
		ph = int(round(h * (pw / w)))
		printer.setPageSize(QtGui.QPageSize(QtCore.QSize(ph, pw), units = QtGui.QPageSize.Point))
		painter = QtGui.QPainter(printer)
		
		self.view.progress.update_state(text = "Rendering...", value = 3, maximum = cmax)
		if self.view.progress.cancel_pressed():
			painter.end()
			os.remove(path)
			return
		
		self.scene().render(painter, source = rect)
		painter.end()
		
		self.view.progress.update_state(text = "Rendering...", value = 4, maximum = cmax)
	
	def on_hover_item(self, item):
		
		if item.node_id in self._labels:
			self.view.statusbar.message(str(self._labels[item.node_id]))
		else:
			self.view.statusbar.message(str(item.node_id))
		
		if item.tool_tip is None:
			self._tool_tip.hide()
		else:
			self._tool_tip.show(item.tool_tip, item)
	
	def on_hover_leave(self):
		
		if self._tool_tip.isVisible():
			if self._tool_tip.underMouse():
				return
			if self._tool_tip.node().isUnderMouse():
				return
			self._tool_tip.hide(TOOLTIP_HIDE_DELAY)
	
	@QtCore.Slot()
	def on_selected(self):
		
		self.view.update()
	
	def on_store_changed(self, *args):
		
		self.clear()
	
	def wheelEvent(self, event):
		
		self.scale_view(2**(event.delta() / 240.0))
	
	def mousePressEvent(self, event):
		
		self._tool_tip.hide()
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
	
	


