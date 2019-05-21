from lib.Slider import (Slider)
from lib.Button import (Button)

from PySide2 import (QtWidgets, QtCore, QtGui)

SLIDERS = [
	["Diameter", 0, 100],
	["Axis", 0, 100],
	["Hamming", 0, 100],
	["Radius", 0, 100],
	["Tangent", 0, 100],
	["Curvature", 0, 100],
]

class WeightsFrame(QtWidgets.QFrame):
	
	def __init__(self, view):
		
		self.view = view
		self.model = view.model
		self.sliders = {}
		
		QtWidgets.QFrame.__init__(self, view)
		
		self.setLayout(QtWidgets.QVBoxLayout())
		self.layout().setContentsMargins(0, 0, 0, 0)
		
		self.group = QtWidgets.QGroupBox("Weights")
		self.group.setLayout(QtWidgets.QVBoxLayout())
		self.slider_frame = QtWidgets.QFrame()
		self.slider_frame.setLayout(QtWidgets.QGridLayout())
		self.slider_frame.layout().setContentsMargins(0, 0, 0, 0)
		self.group.layout().addWidget(self.slider_frame)
		self.layout().addWidget(self.group)
		
		for row in range(len(SLIDERS)):
			name, val_min, val_max = SLIDERS[row]
			self.sliders[name] = Slider(self.view, name, val_min, val_max)
			self.slider_frame.layout().addWidget(QtWidgets.QLabel(name), row, 0)
			self.slider_frame.layout().addWidget(self.sliders[name], row, 1)
		
		self.optimize_button = Button("Optimize", self.view.on_optimize)
		self.group.layout().addWidget(self.optimize_button)
		
		for name in self.model.weights:
			self.set_slider(name, int(round(self.model.weights[name] * 100)))
	
	def set_slider(self, name, value):
		
		if name in self.sliders:
			self.sliders[name].set_value(value)
	
	def update(self):
		
		self.optimize_button.setEnabled(len(self.view.get_selected()) > 1)

	def reload(self):
		
		for name in self.model.weights:
			self.set_slider(name, int(round(self.model.weights[name] * 100)))

