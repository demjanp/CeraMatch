from deposit.utils.fnc_serialize import (try_numeric)

from PySide2 import (QtWidgets, QtCore, QtGui)
from natsort import natsorted
import numbers
import os

class DialogExportDrawing(QtWidgets.QFrame):
	
	def __init__(
		self, dialog, path, formats, 
		multi = False, show_scale = True, show_dpi = True, show_page_size = False
	):
		QtWidgets.QFrame.__init__(self)
		
		self.formats = formats
		self.multi = multi
		self.show_dpi = show_dpi
		self.show_scale = show_scale
		self.show_page_size = show_page_size
		if os.path.isdir(path):
			self.folder = path
		else:
			self.folder = os.path.dirname(path)
		
		self.page_sizes = {}
		for name in natsorted(QtGui.QPageSize.PageSizeId.values):
			self.page_sizes[name] = QtGui.QPageSize.PageSizeId.__dict__[name]
		
		dialog.setModal(True)
		dialog.set_button_box(True, True)
		dialog.set_enabled(True)
		
		self.setMinimumWidth(256)
		self.setLayout(QtWidgets.QFormLayout())
		
		if self.show_scale:
			self.scale_edit = QtWidgets.QLineEdit()
			self.scale_edit.setPlaceholderText("Auto")
			self.scale_edit.setFixedWidth(40)
			self.scale_frame = QtWidgets.QWidget()
			self.scale_frame.setLayout(QtWidgets.QHBoxLayout())
			self.scale_frame.layout().setContentsMargins(0, 0, 0, 0)
			self.scale_frame.layout().addWidget(QtWidgets.QLabel("1:"))
			self.scale_frame.layout().addWidget(self.scale_edit)
			self.scale_frame.layout().addStretch()
		
		if self.show_dpi:
			self.dpi_edit = QtWidgets.QLineEdit("600")
			self.dpi_edit.setFixedWidth(40)
		
		if self.show_page_size:
			self.page_size_combo = QtWidgets.QComboBox()
			self.page_size_combo.setEditable(False)
			items = list(self.page_sizes.keys())
			self.page_size_combo.addItems(items)
			self.page_size_combo.setCurrentIndex(items.index("A4"))
		
		self.linewidth_edit = QtWidgets.QLineEdit("0.2")
		self.linewidth_edit.setFixedWidth(40)
		
		if self.multi and self.formats:
			self.format_combo = QtWidgets.QComboBox()
			self.format_combo.setEditable(False)
			self.format_combo.addItems(list(self.formats.keys()))
		
		self.dest_edit = QtWidgets.QLineEdit()
		self.dest_button = QtWidgets.QPushButton("Browse")
		self.dest_button.clicked.connect(self.on_dest_button)
		self.dest_frame = QtWidgets.QWidget()
		self.dest_frame.setLayout(QtWidgets.QHBoxLayout())
		self.dest_frame.layout().setContentsMargins(0, 0, 0, 0)
		self.dest_frame.layout().addWidget(self.dest_edit)
		self.dest_frame.layout().addWidget(self.dest_button)
		
		if self.show_scale:
			self.layout().addRow("Scale:", self.scale_frame)
		if self.show_dpi:
			self.layout().addRow("DPI:", self.dpi_edit)
		if self.show_page_size:
			self.layout().addRow("Page Size:", self.page_size_combo)
		self.layout().addRow("Line Width:", self.linewidth_edit)
		if self.multi and self.formats:
			self.layout().addRow("Format:", self.format_combo)
		self.layout().addRow("Destination:", self.dest_frame)
		
		if multi:
			self.dest_edit.setText(self.folder)
		else:
			self.dest_edit.setText(path)
	
	@QtCore.Slot()
	def on_dest_button(self):
		
		path = self.dest_edit.text()
		if self.multi:
			path = QtWidgets.QFileDialog.getExistingDirectory(None,
				caption = "Select Destination Folder", dir = self.folder
			)
		else:
			path, _ = QtWidgets.QFileDialog.getSaveFileName(None, 
				"Select Destination File", 
				path, 
				self.formats,
			)
		if path:
			self.dest_edit.setText(os.path.normpath(path))
	
	def get_data(self):
		
		path = self.dest_edit.text().strip()
		scale = 0
		if self.show_scale:
			scale = try_numeric(self.scale_edit.text().strip())
			if not scale:
				scale = 0
		dpi = 0
		if self.show_dpi:
			dpi = try_numeric(self.dpi_edit.text().strip())
		line_width = try_numeric(self.linewidth_edit.text().strip())
		page_size = QtGui.QPageSize.A4
		if self.show_page_size:
			page_size = self.page_sizes[self.page_size_combo.currentText()]
		format = None
		if self.multi and self.formats:
			format = self.formats[self.format_combo.currentText()]
		for value in [scale, dpi, line_width]:
			if not isinstance(value, numbers.Number):
				return None, None, None, None, None
		
		return path, scale, dpi, line_width, format, page_size

