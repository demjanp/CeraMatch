from lib.toolbar._Tool import (Tool)

from PySide2 import (QtWidgets, QtCore, QtGui)
from pathlib import Path
import os

class ExportClustering(Tool):

	def name(self):

		return "Export Clustering"

	def icon(self):

		return "export_xls.svg"

	def help(self):

		return "Export Clustering"

	def enabled(self):

		return self.model.has_samples()

	def triggered(self, state):
		
		def _get_filename():
			
			name = self.model.identifier
			if name:
				name = "%s_clusters" % (name.split("/")[-1].strip("#"))
			if name:
				return name
			return "clusters.xlsx"
		
		format_xlsx = "Excel 2007+ Workbook (*.xlsx)"
		format_csv = "Comma-separated Values (*.csv)"
		formats = ";;".join([format_xlsx, format_csv])
		default_path = os.path.join(str(Path.home()), "Desktop", _get_filename())
		path, format = QtWidgets.QFileDialog.getSaveFileName(None, "Export Clustering As", default_path, formats)
		if not path:
			return
		if format == format_xlsx:
			self.model.clusters.export_xlsx(path)
		elif format == format_csv:
			self.model.clusters.export_csv(path)

