from PySide2 import (QtWidgets, QtCore, QtGui)
from pathlib import Path
import os

class IconThread(QtCore.QThread):
	
	def __init__(self, parent, index, icon_size = 256):
		
		sample = index.data(QtCore.Qt.UserRole)
		
		self.parent = parent
		self.index = index
		self.label = sample.resource
		self.icon_size = icon_size
		self.local_folder = self.parent.model.local_folder
		
		QtCore.QThread.__init__(self)
	
	def run(self):
		
		path = os.path.join(str(Path.home()), "AppData", "Local", "CeraMatch", "thumbnails", "%s.jpg" % (".".join(self.label.filename.split(".")[:-1])))
		if not os.path.isfile(path):
			path = self.parent.model.images.get_thumbnail(self.label, size = self.icon_size, root_folder = self.local_folder)
		self.parent.on_icon_thread(self.index, path)
