from ceramatch.view.vcontrols_groups.descriptors import Descriptors
from ceramatch.view.vcontrols_groups.distances import Distances
from ceramatch.view.vcontrols_groups.clusters import Clusters
from ceramatch.view.vcontrols_groups.attributes import Attributes
from ceramatch.view.vertical_scroll_area import VerticalScrollArea

from deposit.utils.fnc_files import (as_url, url_to_path)

from PySide2 import (QtWidgets, QtCore, QtGui)

class VControls(QtWidgets.QFrame):
	
	signal_folder_link_clicked = QtCore.Signal(str)		# path
	
	def __init__(self):
		
		QtWidgets.QFrame.__init__(self)
		
		self.descriptors = Descriptors()
		self.distances = Distances()
		self.clusters = Clusters()
		self.attributes = Attributes()
		
		self.setLayout(QtWidgets.QVBoxLayout())
		self.layout().setContentsMargins(10, 10, 0, 10)
		self.setSizePolicy(
			QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum
		)
		self.setMaximumWidth(400)
		
		self._dbname_label = QtWidgets.QLabel("Database: ")
		self._folder_label = QtWidgets.QLabel("Local Folder: ")
		self._folder_label.setWordWrap(True)
		self._folder_label.linkActivated.connect(self.on_folder_link)
		
		frame_db = QtWidgets.QFrame()
		frame_db.setLayout(QtWidgets.QVBoxLayout())
		frame_db.layout().setContentsMargins(5, 5, 5, 5)
		frame_db.setStyleSheet(
			".QFrame {border-style: solid; border-width: 1px; border-color: grey;}"
		)
		frame_db.layout().addWidget(self._dbname_label)
		frame_db.layout().addWidget(self._folder_label)
		
		frame_groups = QtWidgets.QFrame()
		frame_groups.setLayout(QtWidgets.QVBoxLayout())
		frame_groups.layout().setContentsMargins(0, 0, 0, 0)
		frame_groups.layout().addWidget(self.descriptors)
		frame_groups.layout().addWidget(self.distances)
		frame_groups.layout().addWidget(self.clusters)
		frame_groups.layout().addWidget(self.attributes)
		
		scroll_area = VerticalScrollArea(frame_groups)
		
		self.layout().addWidget(frame_db)
		self.layout().addWidget(scroll_area)
#		self.layout().addStretch()
	
	@QtCore.Slot(str)
	def on_folder_link(self, url):
		
		self.signal_folder_link_clicked.emit(url_to_path(url))
	
	def set_db_name(self, name):
		
		self._dbname_label.setText("Database: <b>%s</b>" % (name))
	
	def set_folder(self, path, url = None, max_path_length = 10):
		
		if path:
			if url is None:
				url = as_url(path)
			if len(path) > max_path_length:
				path = "\\ ".join(path.split("\\"))
			self._folder_label.setText("Local Folder: <a href=\"%s\">%s</a>" % (url, path))
		else:
			self._folder_label.setText("Local Folder: <b>not set</b>")
