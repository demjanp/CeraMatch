from lib.Button import (Button)

from PySide2 import (QtWidgets, QtCore, QtGui)

class FooterFrame(QtWidgets.QFrame):
	
	def __init__(self, view):
		
		self.view = view
		self.model = view.model
		
		QtWidgets.QFrame.__init__(self, view)
		
		self.setLayout(QtWidgets.QHBoxLayout())
		
		self.slider_zoom = QtWidgets.QSlider(QtCore.Qt.Horizontal)
		self.slider_zoom.setMinimum(64)
		self.slider_zoom.setMaximum(256)
		self.slider_zoom.valueChanged.connect(self.view.on_zoom)
		
		self.reload_button = Button("Reload", self.view.on_reload)
		self.prev_button = Button("Previous", self.view.on_prev)
		self.next_button = Button("Next", self.view.on_next)
		
		self.layout().addWidget(self.reload_button)
		self.layout().addWidget(self.prev_button)
		self.layout().addWidget(self.next_button)
		self.layout().addStretch()
		self.layout().addWidget(QtWidgets.QLabel("Zoom:"))
		self.layout().addWidget(self.slider_zoom)
	
	def update(self):
		
		browse_enabled = (self.view.mode == self.view.MODE_DISTANCE)
		self.prev_button.setEnabled(browse_enabled)
		self.next_button.setEnabled(browse_enabled)
