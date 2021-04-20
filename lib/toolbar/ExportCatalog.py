from lib.toolbar._Tool import (Tool)

from PySide2 import (QtWidgets, QtCore, QtGui)
from pathlib import Path
import os

class ExportCatalog(Tool):

	def name(self):

		return "Export Catalog"
	
	def icon(self):

		return "export_catalog.svg"
	
	def help(self):

		return "Export Catalog as PDF"
	
	def enabled(self):

		return self.model.has_clusters()
	
	def triggered(self, state):
		
		def _get_filename():
			
			name = self.model.identifier
			if name:
				name = "%s_catalog.pdf" % (name.split("/")[-1].strip("#"))
			if name:
				return name
			return "catalog.pdf"
		
		default_path = os.path.join(str(Path.home()), "Desktop", _get_filename())
		path, _ = QtWidgets.QFileDialog.getSaveFileName(None, "Export Catalog As", default_path, "Adobe PDF (*.pdf)")
		if not path:
			return
		self.view.save_catalog(path)


