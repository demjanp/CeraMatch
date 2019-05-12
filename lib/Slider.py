from PySide2 import (QtWidgets, QtCore, QtGui)

class Slider(QtWidgets.QFrame):
	
	def __init__(self, view, name, val_min, val_max):
		
		self.view = view
		self.model = view.model
		self.name = name
		self.val_min = val_min
		self.val_max = val_max
		
		QtWidgets.QFrame.__init__(self)
		
		self.setLayout(QtWidgets.QHBoxLayout())
		self.layout().setContentsMargins(0, 0, 0, 0)
		
		self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
		self.slider.setMinimum(self.val_min)
		self.slider.setMaximum(self.val_max)
		
		self.edit = QtWidgets.QLineEdit()
		self.edit.setMaximumWidth(50)
		self.edit.setText(str(self.val_min))
		
		self.layout().addWidget(self.slider)
		self.layout().addWidget(self.edit)
		
		self.slider.valueChanged.connect(self.on_value_changed)
		self.edit.textChanged.connect(self.on_value_changed)
	
	def get_value(self):
		
		return(self.slider.value())
	
	def set_value(self, value):
		
		self.slider.blockSignals(True)
		self.edit.blockSignals(True)
		self.slider.setValue(value)
		self.edit.setText(str(value))
		self.slider.blockSignals(False)
		self.edit.blockSignals(False)
	
	def set_maximum(self, value):
		
		self.val_max = value
		self.slider.setMaximum(self.val_max)
	
	def on_value_changed(self, value):
		
		try:
			value = int(value)
		except:
			return
		self.set_value(value)
		self.view.on_slider(self.name, value)

