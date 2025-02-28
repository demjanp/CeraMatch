from deposit_gui import DCActions

from deposit.utils.fnc_files import (as_url, sanitize_filename)

class CActions(DCActions):
	
	def __init__(self, cmain, cview) -> None:
		
		DCActions.__init__(self, cmain, cview)
		
		self.update()
	
	# ---- Signal handling
	# ------------------------------------------------------------------------
	
	
	# ---- get/set
	# ------------------------------------------------------------------------
	
	
	# ---- Actions
	# ------------------------------------------------------------------------
	def set_up_tool_bar(self):
		
		return {
			"Data": [
				("Connect", "Connect"),
				("Save", "Save"),
				("Deposit", "Open Deposit"),
				None,
				("ImportClustering", "Import Clustering"),
				("ExportClustering", "Export Clustering"),
			],
			"Edit": [
				("Undo", "Undo"),
				("Redo", "Redo"),
				None,
				("SelectDescendants", "Select Descendants"),
			],
			"Export": [
				("ExportDendrogram", "Export Dendrogram as PDF"),
				("ExportCatalog", "Export Catalog as PDF"),
			],
		}
	
	def set_up_menu_bar(self):
		
		return {
			"Data": [
				("Connect", "Connect"),
				("Save", "Save"),
				("SaveAsFile", "Save As File"),
				("SaveAsPostgres", "Save As PostgreSQL"),
				("Deposit", "Open Deposit"),
				None,
				("AutoBackup", "Backup database after every save"),
				None,
				("ImportClustering", "Import Clustering"),
				("ExportClustering", "Export Clustering"),
			],
			"Edit": [
				("Undo", "Undo"),
				("Redo", "Redo"),
				None,
				("SelectDescendants", "Select Descendants"),
			],
			"Export": [
				("ExportDendrogram", "Export Dendrogram"),
				("ExportCatalog", "Export Catalog"),
			],
			"Help": [
				("About", "About"),
				("LogFile", "Log File"),
			],
		}
	
	
	# implement update_[name] and on_[name] for each action
	'''
	def update_[name](self):
		
		return dict(
			caption = "Caption",
			icon = "icon.svg",
			shortcut = "Ctrl+S",
			help = "Tool tip",
			combo: list,
			checkable = True,
			checked = True,
			enabled = True,
		)
	
	def on_[name](self, state):
		
		pass
	'''
	
	def update_Connect(self):
		
		return dict(
			caption = "Connect",
			icon = "connect.svg",
			help = "Connect to Database",
			checkable = False,
			enabled = True,
		)
	
	def on_Connect(self, state):
		
		self.cmain.cdialogs.open("Connect")
	
	
	def update_Save(self):
		
		return dict(
			caption = "Save",
			icon = "save.svg",
			help = "Save",
			checkable = False,
			enabled = not self.cmain.cmodel.is_saved(),
		)
	
	def on_Save(self, state):
		
		if self.cmain.cmodel.can_save():
			self.cmain.cmodel.save()
		else:
			self.on_SaveAsFile(True)
	
	def on_SaveAsFile(self, state):
		
		path, format = self.cmain.cview.get_save_path("Save Database As", "Pickle (*.pickle);;JSON (*.json)")
		if not path:
			return
		self.cmain.cview.set_recent_dir(path)
		self.cmain.cmodel.save(path = path)
		url = as_url(path)
		self.cmain.cdialogs.open("ConfirmLoad", url)
	
	
	def update_SaveAsPostgres(self):
		
		return dict(
			help = "Save As PostgreSQL Database",
			checkable = False,
			enabled = True,
		)
	
	def on_SaveAsPostgres(self, state):
		
		self.cmain.cdialogs.open("SaveAsPostgres")
	
	
	def update_Deposit(self):
		
		return dict(
			icon = "deposit.svg",
			help = "Open Deposit",
			checkable = False,
			enabled = True,
		)
	
	def on_Deposit(self, state):
		
		self.cmain.open_deposit()
	
	
	def update_ImportClustering(self):
		
		return dict(
			icon = "import_xls.svg",
			help = "Import Clustering",
			checkable = False,
			enabled = True,
		)
	
	def on_ImportClustering(self, state):
		
		self.cmain.cdialogs.open("ImportClustering")
	
	
	def update_ExportClustering(self):
		
		return dict(
			icon = "export_xls.svg",
			help = "Export Clustering",
			checkable = False,
			enabled = self.cmain.cgraph.has_clusters(),
		)
	
	def on_ExportClustering(self, state):
		
		filename = sanitize_filename(
			self.cmain.cmodel.get_datasource_name() + "_clusters.xlsx"
		)
		path, format = self.cmain.cview.get_save_path(
			"Export Clustering As",
			"Excel 2007+ Workbook (*.xlsx);;Comma-separated Values (*.csv)",
			filename = filename
		)
		if path is None:
			return
		self.cmain.cgraph.export_clusters(path, format)
	
	
	def update_Undo(self):
		
		return dict(
			caption = "Undo",
			icon = "undo.svg",
			help = "Undo last action (Ctrl+Z)",
			shortcut = "Ctrl+Z",
			checkable = False,
			enabled = self.cmain.history.can_undo(),
		)
	
	def on_Undo(self, state):
		
		self.cmain.history.undo()
	
	
	def update_Redo(self):
		
		return dict(
			caption = "Redo",
			icon = "redo.svg",
			help = "Redo last undone action (Ctrl+Y)",
			shortcut = "Ctrl+Y",
			checkable = False,
			enabled = self.cmain.history.can_redo(),
		)
	
	def on_Redo(self, state):
		
		self.cmain.history.redo()
	
	
	def update_SelectDescendants(self):
		
		return dict(
			caption = "Select Descendants (Ctrl+D)",
			icon = "select_descendants.svg",
			help = "Select Descendants of Node (Ctrl+D)",
			shortcut = "Ctrl+D",
			checkable = False,
			enabled = len(self.cmain.cgraph.get_selected()) > 0,
		)
	
	def on_SelectDescendants(self, state):
		
		self.cmain.cgraph.select_descendants()
	
	
	def update_ExportDendrogram(self):
		
		return dict(
			icon = "export_dendrogram.svg",
			help = "Export Dendrogram as PDF",
			checkable = False,
			enabled = self.cmain.cgraph.has_drawings(),
		)
	
	def on_ExportDendrogram(self, state):
		
		self.cmain.cdialogs.open("ExportDendrogram")
	
	
	def update_ExportCatalog(self):
		
		return dict(
			icon = "export_catalog.svg",
			help = "Export Clustered Drawings as PDF Catalog",
			checkable = False,
			enabled = self.cmain.cgraph.has_clusters(),
		)
	
	def on_ExportCatalog(self, state):
		
		self.cmain.cdialogs.open("ExportCatalog")
	
	
	def update_AutoBackup(self):
		
		return dict(
			help = "Backup database after every save",
			checkable = True,
			checked = self.cmain.cmodel.has_auto_backup(),
			enabled = True,
		)
	
	def on_AutoBackup(self, state):
		
		self.cmain.cmodel.set_auto_backup(state)
	
	
	def on_About(self, state):
		
		self.cmain.cdialogs.open("About")
	
	def on_LogFile(self, state):
		
		self.cmain.open_in_external(self.cmain.cview.get_logging_path())

