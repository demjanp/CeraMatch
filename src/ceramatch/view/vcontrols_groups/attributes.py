from ceramatch.view.vcontrols_groups.controls.button import (Button)

from PySide2 import (QtWidgets, QtCore, QtGui)
from deposit.utils.fnc_files import (as_url, url_to_path)

class AttributeControl(object):
	
	signal_changed = QtCore.Signal(str, str)  # (name, value)
	
	def __init__(self, name):
		
		self.name = name
	
	def get_value(self):
		
		return ""
	
	def set_value(self, value):
		
		pass
	
	def set_items(self, values):
		
		pass
	
	@QtCore.Slot()
	def on_changed(self, *args):
		
		self.signal_changed.emit(self.name, self.get_value())

class LineEdit(AttributeControl, QtWidgets.QLineEdit):
	
	def __init__(self, name):
		
		AttributeControl.__init__(self, name)
		QtWidgets.QLineEdit.__init__(self)
		
		self.textChanged.connect(self.on_changed)
	
	def get_value(self):
		
		return self.text().strip()
	
	def set_value(self, value):
		
		self.blockSignals(True)
		self.setText(value)
		self.blockSignals(False)

class ComboBox(AttributeControl, QtWidgets.QComboBox):
	
	def __init__(self, name):
		
		AttributeControl.__init__(self, name)
		QtWidgets.QComboBox.__init__(self)
		
		self.setEditable(True)
		
		self.currentTextChanged.connect(self.on_changed)
	
	def get_value(self):
		
		return self.currentText().strip()
	
	def set_value(self, value):
		
		self.blockSignals(True)
		self.setCurrentText(value)
		self.blockSignals(False)
	
	def set_items(self, values):
		
		self.blockSignals(True)
		current_value = self.currentText()
		self.clear()
		if values:
			self.addItems(values)
		if current_value in values:
			self.setCurrentIndex(values.index(current_value))
		elif current_value:
			self.setCurrentText(current_value)
		self.blockSignals(False)

class CheckBox(AttributeControl, QtWidgets.QCheckBox):
	
	def __init__(self, name):
		
		
		AttributeControl.__init__(self, name)
		QtWidgets.QCheckBox.__init__(self)
		
		self.stateChanged.connect(self.on_changed)
	
	def get_value(self):
		
		return str(int(self.isChecked()))
	
	def set_value(self, value):
		
		value_ = False
		try:
			value_ = bool(int(value))
		except:
			raise Exception(
				"Error: Could not convert attribute %s, value '%s' to bool" % (self.name, str(value))
			)
		self.blockSignals(True)
		self.setChecked(value_)
		self.blockSignals(False)
	
	def set_items(self, values):
		
		pass

CONTROL_CLASSES = {
	"LineEdit": LineEdit,
	"ComboBox": ComboBox,
	"CheckBox": CheckBox,
}

class Attributes(QtWidgets.QGroupBox):
	
	signal_store_attributes = QtCore.Signal()
	
	def __init__(self):
		
		QtWidgets.QGroupBox.__init__(self, "Attributes")
		
		self._attributes = {} #  {lap_name: control, ...}
		self._labels = {}  # {lap_name: label, ...}
		self._attributes_changed = False
		
		self.setStyleSheet("QGroupBox {font-weight: bold;}")
		self.setLayout(QtWidgets.QVBoxLayout())
		
		self._form_frame = QtWidgets.QFrame()
		self._form_frame.setLayout(QtWidgets.QFormLayout())
		self._form_frame.layout().setContentsMargins(5, 5, 5, 5)
		
		self.store_button = Button("Store", self.on_store)
		
		label = QtWidgets.QLabel(
			"To modify attributes, use the LAP application - 'Edit Descriptors' function."
		)
		label.setWordWrap(True)
		
		self.layout().addWidget(self._form_frame)
		self.layout().addWidget(self.store_button)
		self.layout().addWidget(label)
		
		self.update_store_button()
	
	
	# ---- Signal handling
	# ------------------------------------------------------------------------
	@QtCore.Slot(str, str)
	def on_changed(self, name, value):
		
		self._attributes_changed = True
		self.update_store_button()
	
	@QtCore.Slot(str, str)
	def on_store(self):
		
		self.signal_store_attributes.emit()
	
	# ---- get/set
	# ------------------------------------------------------------------------
	def update_store_button(self):
		
		state = False
		if self._attributes_changed and self.get_data():
			state = True
		self.store_button.setEnabled(state)
	
	def populate(self, rows):
		# rows = [(label, ctrl_type, name), ...]
		
		self._attributes_changed = False
		self._attributes = {}
		self._labels = {}
		layout = self._form_frame.layout()
		for row in reversed(range(layout.rowCount())):
			layout.removeRow(row)
		for label, ctrl_type, name in rows:
			if ctrl_type not in CONTROL_CLASSES:
				continue
			self._attributes[name] = CONTROL_CLASSES[ctrl_type](name)
			self._attributes[name].signal_changed.connect(self.on_changed)
			self._labels[name] = label
			layout.addRow(QtWidgets.QLabel("%s:" % label), self._attributes[name])
		self.update_store_button()
	
	def clear(self):
		
		self.populate([])
		self.update_store_button()
	
	def get_data(self):
		# return data = {name: value, ...}
		
		data = {}
		for name in self._attributes:
			value = self._attributes[name].get_value()
			if value:
				data[name] = value
		
		return data
	
	def set_data(self, data):
		# data = {name: (value, items), ...}; items = [value, ...]
		
		self._attributes_changed = False
		for name in self._attributes:
			value, items = data.get(name, ("", []))
			if value is None:
				value = ""
			self._attributes[name].set_items([str(value) for value in items])
			self._attributes[name].set_value(str(value))
		self.update_store_button()
