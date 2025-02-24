from ceramatch.dialogs.dialog_connect import DialogConnect
from ceramatch.dialogs.dialog_import_clustering import DialogImportClustering
from ceramatch.dialogs.dialog_export_drawing import DialogExportDrawing
from ceramatch.dialogs.dialog_about import DialogAbout

from ceramatch.utils.fnc_matching import (calc_distances)
from deposit.utils.fnc_files import (sanitize_filename)

from deposit_gui import DCDialogs

from PySide6 import (QtWidgets, QtCore, QtGui)
import os

class CDialogs(DCDialogs):
	
	def __init__(self, cmain, cview):
		
		DCDialogs.__init__(self, cmain, cview)
		
		self.view = cview._view
	
	# ---- Signal handling
	# ------------------------------------------------------------------------
	@QtCore.Slot()
	def on_clear_recent(self):
		
		self.view.clear_recent_connections()
	
	
	# ---- get/set
	# ------------------------------------------------------------------------
	
	
	# ---- Dialogs
	# ------------------------------------------------------------------------
	'''
	open(name, *args, **kwargs)
	
	Implement set_up_[name], process_[name] and cancel_[name] for each dialog:
	
	def set_up_[name](self, dialog, *args, **kwargs):
		
		args and kwargs are passed from DCDialogs.open(name, *args, **kwargs)
		
		dialog = QtWidgets.QDialog
		
		dialog.set_title(name)
		dialog.set_frame(frame = QtWidget)
		dialog.get_frame()
		dialog.set_button_box(ok: bool, cancel: bool): set if OK and Cancel buttons are visible
		
	def process_[name](self, dialog, *args, **kwargs):
		
		args and kwargs are passed from DCDialogs.open(name, *args, **kwargs)
		
		process dialog after OK has been clicked
	
	def cancel_[name](self, dialog, *args, **kwargs):
		
		args and kwargs are passed from DCDialogs.open(name, *args, **kwargs)
		
		handle dialog after cancel has been clicked
	'''
	
	def set_up_Connect(self, dialog, *args, **kwargs):
		
		frame = DialogConnect(dialog)
		frame.set_recent_dir(self.view.get_recent_dir())
		frame.set_recent_connections(self.view.get_recent_connections())
		frame.signal_clear_recent.connect(self.on_clear_recent)
	
	def process_Connect(self, dialog, *args, **kwargs):
		
		self.cmain.cmodel.load(**dialog.get_data())
	
	def cancel_Connect(self, dialog, *args, **kwargs):
		
		datasource = self.cmain.cmodel.get_datasource()
		if not datasource.is_valid():
			self.cmain.cmodel.load(datasource = "Memory")
	
	
	def set_up_SaveAsPostgres(self, dialog, *args, **kwargs):
		
		frame = DSaveAsPostgresFrame(dialog)
		frame.set_recent_connections(self.view.get_recent_connections())
	
	def process_SaveAsPostgres(self, dialog, *args, **kwargs):
		
		data = dialog.get_data()
		datasource = data["datasource"]
		self.cmain.cmodel.set_local_folder(datasource.get_local_folder())
		self.cmain.cmodel.save(**data)
		
		if self.cmain.cview.show_question(
			"Load Database?", 
			"Load database from <b>%s</b>" % (datasource.get_name()),
		):
			self.cmain.cmodel.load(**data)
	
	
	def set_up_SelectDistances(self, dialog):
		
		dialog.set_title("Select Distances to Calculate")
		dialog.setModal(True)
		dialog.set_button_box(True, True)
		
		frame = QtWidgets.QFrame()
		frame.setMinimumWidth(350)
		frame.setLayout(QtWidgets.QVBoxLayout())
		frame.check_diam = QtWidgets.QCheckBox("Diameter")
		frame.check_diam.setChecked(True)
		frame.check_ax = QtWidgets.QCheckBox("Axis")
		frame.check_ax.setChecked(True)
		frame.check_dice = QtWidgets.QCheckBox("Dice")
		frame.check_dice.setChecked(True)
		frame.check_dice_rim = QtWidgets.QCheckBox("Dice (rim only)")
		frame.check_dice_rim.setChecked(True)
		
		warning_frame = QtWidgets.QFrame()
		warning_frame.setLayout(QtWidgets.QHBoxLayout())
		icon = QtWidgets.QApplication.style().standardIcon(
			QtWidgets.QStyle.SP_MessageBoxWarning
		)
		label = QtWidgets.QLabel()
		label.setPixmap(icon.pixmap(QtCore.QSize(32, 32)))
		warning_frame.layout().addWidget(label)
		label = QtWidgets.QLabel(
			"Warning: Dice distance calculation is very computationally intensive"
		)
		label.setWordWrap(True)
		warning_frame.layout().addWidget(label)
		warning_frame.layout().addStretch()
		
		frame.layout().addWidget(frame.check_diam)
		frame.layout().addWidget(frame.check_ax)
		frame.layout().addWidget(frame.check_dice)
		frame.layout().addWidget(frame.check_dice_rim)
		frame.layout().addWidget(warning_frame)
		
		dialog.set_frame(frame)
	
	def process_SelectDistances(self, dialog):
		
		frame = dialog.get_frame()
		diam = frame.check_diam.isChecked()
		axis = frame.check_ax.isChecked()
		dice = frame.check_dice.isChecked()
		dice_rim = frame.check_dice_rim.isChecked()
		
		data = self.cmain.cgraph.get_profile_data()
		# data = {obj_id: (coords, radius), ...}
		
		select_components = [diam, axis, dice, dice_rim]
		calc_distances(
			data, select_components, self.cmain.cmodel._distance,
			progress = self.cmain.cview.progress,
		)
		
		self.cmain.cmodel.save_distance()
	
	
	def set_up_ImportClustering(self, dialog):
		
		dialog.set_title("")
		dialog.set_button_box(True, True)
		dialog.setModal(True)
		dialog.set_frame(DialogImportClustering(self.cmain.cview))
	
	def process_ImportClustering(self, dialog):
		
		frame = dialog.get_frame()
		path = frame.path_edit.text().strip()
		sample_column = frame.sample_combo.currentText()
		cluster_column = frame.cluster_combo.currentText()
		if (not sample_column) or (not cluster_column) or (not os.path.isfile(path)):
			return
		self.cmain.cgraph.import_clusters(path, sample_column, cluster_column)
	
	
	def set_up_ExportDendrogram(self, dialog):
		
		dialog.set_title("Export PDF Dendrogram")
		path = self.cmain.cview.get_recent_dir()
		if not path:
			path = self.cmain.cview.get_default_folder()
		filename = "%s.pdf" % sanitize_filename(
			self.cmain.cmodel.get_datasource_name() + "_dendrogram",
			"dendrogram",
		)
		frame = DialogExportDrawing(
			dialog,
			path = os.path.normpath(os.path.join(path, filename)),
			formats = "Adobe PDF (*.pdf)",
			multi = False,
			show_scale = False,
			show_dpi = True,
			show_page_size = True,
		)
		dialog.set_frame(frame)
	
	def process_ExportDendrogram(self, dialog):
		
		path, _, dpi, line_width, _, page_size = dialog.get_frame().get_data()
		if path is None:
			return
		
		self.cmain.cgraph.export_dendrogram(
			path, dpi, page_size, stroke_width = line_width
		)
		self.cmain.cview.set_recent_dir(path)
	
	
	def set_up_ExportCatalog(self, dialog):
		
		dialog.set_title("Export PDF Catalog")
		path = self.cmain.cview.get_recent_dir()
		if not path:
			path = self.cmain.cview.get_default_folder()
		filename = "%s.pdf" % sanitize_filename(
			self.cmain.cmodel.get_datasource_name() + "_catalog",
			"catalog",
		)
		frame = DialogExportDrawing(
			dialog,
			path = os.path.normpath(os.path.join(path, filename)),
			formats = "Adobe PDF (*.pdf)",
			multi = False,
			show_scale = True,
			show_dpi = False,
			show_page_size = True,
		)
		dialog.set_frame(frame)
	
	def process_ExportCatalog(self, dialog):
		
		path, scale, _, line_width, _, page_size = dialog.get_frame().get_data()
		if path is None:
			return
		
		if scale > 0:
			scale = 1 / scale
		else:
			scale = 1 / 3
		self.cmain.cgraph.export_catalog(
			path, scale, page_size, stroke_width = line_width
		)
		self.cmain.cview.set_recent_dir(path)
	
	
	def set_up_About(self, dialog):
		
		dialog.set_title("About CeraMatch")
		dialog.set_button_box(True, False)
		dialog.setModal(True)
		dialog.set_frame(DialogAbout())

