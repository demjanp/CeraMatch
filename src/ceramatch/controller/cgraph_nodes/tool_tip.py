from PySide6 import (QtWidgets, QtCore, QtGui)
import weakref

TOOLTIP_SHOW_DELAY = 800
TOOLTIP_HIDE_DELAY = 400

class ToolTip(QtWidgets.QLabel):
	
	def __init__(self):
		
		self._node = None
		
		QtWidgets.QLabel.__init__(self)
		
		self.setWindowFlags(QtCore.Qt.ToolTip)
		self.setFrameShape(QtWidgets.QFrame.StyledPanel)
		self.setStyleSheet("QLabel { background-color : white; }")
		
		self._hide_timer = QtCore.QTimer(singleShot = True, timeout = self.on_hide_timer)
		self._show_timer = QtCore.QTimer(singleShot = True, timeout = self.on_show_timer)
		
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
		
		self._show_timer.stop()
		self._hide_timer.stop()
		self._node = weakref.ref(node)
		self.setText(text)
		self.adjustSize()
		self.move(QtGui.QCursor.pos())
		self._show_timer.start(TOOLTIP_SHOW_DELAY)
	
	def hide(self, delay = None):
		
		self._show_timer.stop()
		if delay is None:
			self._hide_timer.stop()
			QtWidgets.QLabel.hide(self)
			return
		if self.underMouse():
			return
		self._hide_timer.start(delay)
	
	@QtCore.Slot()
	def on_show_timer(self):
		
		node = self._node()
		if node is None:
			return
		if QtWidgets.QApplication.mouseButtons() != QtCore.Qt.NoButton:
			return
		if node.isUnderMouse():
			QtWidgets.QLabel.show(self)
	
	@QtCore.Slot()
	def on_hide_timer(self):
		
		self.hide()
	
	def leaveEvent(self, event):
		
		if self.isVisible():
			self.hide(TOOLTIP_HIDE_DELAY)
		else:
			self._show_timer.stop()
		
		QtWidgets.QLabel.leaveEvent(self, event)
	
	def mousePressEvent(self, event):
		
		self.hide()
		
		QtWidgets.QLabel.mousePressEvent(self, event)
