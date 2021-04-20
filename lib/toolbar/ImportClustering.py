from lib.toolbar._Tool import (Tool)

class ImportClustering(Tool):

	def name(self):

		return "Import Clustering"

	def icon(self):

		return "import_xls.svg"

	def help(self):

		return "Import Clustering"

	def enabled(self):

		return self.model.has_samples()

	def triggered(self, state):
		
		self.view.dialogs.open("Import")

