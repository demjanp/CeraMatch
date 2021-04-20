from lib.toolbar._Tool import (Tool)

from PySide2 import (QtWidgets, QtCore, QtGui)
from pathlib import Path
import os

class ExportDendrogram(Tool):

	def name(self):

		return "Export Dendrogram"
	
	def icon(self):

		return "export_dendrogram.svg"
	
	def help(self):

		return "Export Dendrogram as PDF"
	
	def enabled(self):

		return True
	
	def triggered(self, state):
		
		def _get_filename():
			
			name = self.model.identifier
			if name:
				name = "%s_dendrogram.pdf" % (name.split("/")[-1].strip("#"))
			if name:
				return name
			return "dendrogram.pdf"
		
		default_path = os.path.join(str(Path.home()), "Desktop", _get_filename())
		path, _ = QtWidgets.QFileDialog.getSaveFileName(None, "Export Dendrogram As", default_path, "Adobe PDF (*.pdf)")
		if not path:
			return
		self.view.save_dendrogram(path)


