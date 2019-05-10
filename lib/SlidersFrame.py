from PySide2 import (QtWidgets, QtCore, QtGui)

SLIDERS = [
	[0, 0, "Radius", 0, 99],
	[1, 0, "Tangent", 0, 99],
	[2, 0, "Curvature", 0, 99],
	[3, 0, "Hamming", 0, 99],
	[4, 0, "Diameter", 0, 99],
	[5, 0, "Axis", 0, 99],
]

class Slider(QtWidgets.QSlider):
	
	def __init__(self, view, name, val_min, val_max):
		
		self.view = view
		self.model = view.model
		self.name = name
		self.val_min = val_min
		self.val_max = val_max
		
		QtWidgets.QSlider.__init__(self, QtCore.Qt.Horizontal)
		
		self.setMinimum(self.val_min)
		self.setMaximum(self.val_max)
		
		self.valueChanged.connect(self.on_value_changed)
	
	def on_value_changed(self, *args):
		
		self.view.on_slider(self.name, self.value())

class SlidersFrame(QtWidgets.QFrame):
	
	def __init__(self, view):
		
		self.view = view
		self.model = view.model
		self.sliders = {}
		
		QtWidgets.QFrame.__init__(self, view)
		
		self.setLayout(QtWidgets.QGridLayout())
		
		for row, col, name, val_min, val_max in SLIDERS:
			self.sliders[name] = Slider(self.view, name, val_min, val_max)
			self.layout().addWidget(QtWidgets.QLabel(name), row, col * 4)
			self.layout().addWidget(QtWidgets.QLabel("-"), row, col * 4 + 1)
			self.layout().addWidget(self.sliders[name], row, col * 4 + 2)
			self.layout().addWidget(QtWidgets.QLabel("+"), row, col * 4 + 3)
		
		for name in self.model.weights:
			self.set_slider(name, int(round(self.model.weights[name] * 100)))
	
	def set_slider(self, name, value):
		
		if name in self.sliders:
			self.sliders[name].blockSignals(True)
			self.sliders[name].setValue(value)
			self.sliders[name].blockSignals(False)

