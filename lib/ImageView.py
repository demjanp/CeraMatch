from lib.ImageList import (ImageList)
from lib.ImageTable import (ImageTable)

from PySide2 import (QtWidgets, QtCore, QtGui)

class ImageView(QtWidgets.QTabWidget):
	
	def __init__(self, view):
		
		self.view = view
		self.model = view.model
		
		QtWidgets.QTabWidget.__init__(self)
		
		self.tab_list = ImageList(self.view)
		self.tab_table = ImageTable(self.view)
		
		self.addTab(self.tab_list, "List")
		self.addTab(self.tab_table, "Clusters")
		
		self.currentChanged.connect(self.on_tab_changed)
	
	def is_list(self):
		
		return self.currentIndex() == 0
	
	def get_current(self):
		
		return [self.tab_list, self.tab_table][self.currentIndex()]
	
	def reload(self):
		
		self.tab_list.reload()
		self.tab_table.reload()
	
	def update_(self):
		
		self.tab_list.update_()
		self.tab_table.update_()
		self.setTabEnabled(1, self.model.has_clusters())
	
	def set_thumbnail_size(self, value):
		
		self.tab_list.set_thumbnail_size(value)
		self.tab_table.set_thumbnail_size(value)
	
	def set_selected(self, sample_ids):
		
		self.tab_list.set_selected(sample_ids)
		self.tab_table.set_selected(sample_ids)
	
	def on_tab_changed(self, index):
		
		if (index == 1) and (self.view.mode != self.view.MODE_CLUSTER):
			self.view.on_cluster()
		self.view.update()
	
	def __getattr__(self, attr):
		
		current = self.get_current()
		if hasattr(current, attr):
			return getattr(current, attr)
		return QtWidgets.QTabWidget.__getattr__(self, attr)
