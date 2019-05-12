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
		
		self.layout().addStretch()
		self.layout().addWidget(QtWidgets.QLabel("Zoom:"))
		self.layout().addWidget(self.slider_zoom)
		