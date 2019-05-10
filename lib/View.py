from lib.Model import (Model)
from lib.ImageList import (ImageList)
from lib.ButtonsFrame import (ButtonsFrame)
from lib.SlidersFrame import (SlidersFrame)
from lib.StatusBar import (StatusBar)

from PySide2 import (QtWidgets, QtCore, QtGui)
import os

class View(QtWidgets.QMainWindow):

	def __init__(self):
		
		self.model = None
		self.registry_dc = None
		self._distance_mode = False
		self._last_selected = None
		self._loaded = False
		
		QtWidgets.QMainWindow.__init__(self)
		
		self.model = Model(self)
		
		self.setWindowTitle("CeraMatch")
		self.setStyleSheet("font-size: 11pt;")
		
		self.central_widget = QtWidgets.QWidget(self)
		self.central_widget.setLayout(QtWidgets.QVBoxLayout())
		self.central_widget.layout().setContentsMargins(0, 0, 0, 0)
		self.setCentralWidget(self.central_widget)
		
		self.image_lst = ImageList(self)
		self.buttons_frame = ButtonsFrame(self)
		self.sliders_frame = SlidersFrame(self)
		self.statusbar = StatusBar(self)
		
		self.central_widget.layout().addWidget(self.image_lst)
		self.central_widget.layout().addWidget(self.buttons_frame)
		self.central_widget.layout().addWidget(self.sliders_frame)
		self.setStatusBar(self.statusbar)
		
		self._loaded = True
		
		self.setGeometry(100, 100, 1024, 768)
	
	def distance_mode_on(self):
		
		return self._distance_mode
	
	def update(self):
		
		if not self._loaded:
			return
		self.buttons_frame.update()
		
		selected = self.image_lst.get_selected()
		if selected:
			sample_id, _, _, sort_label, _ = selected[0]
			self.statusbar.message("Label: %s, Sample ID: %s" % (sort_label, sample_id))
	
	def on_distance_mode(self, *args):
		
		if not self._distance_mode:
			self._last_selected = None
			selected = self.image_lst.get_selected()
			if selected:
				self._last_selected = selected[0][0]
			if self._last_selected:
				self._distance_mode = True
		
		self.update()
		self.on_recalc()
	
	def on_clustering_mode(self, *args):
		
		if self._distance_mode:
			self._distance_mode = False
		self.update()
		self.on_recalc()
	
	def on_slider(self, name, value):
		
		if name in self.model.weights:
			self.model.set_weight(name, value / 100)
	
	def on_recalc(self, *args):
		
		if self.distance_mode_on():
			selected = self.image_lst.get_selected()
			if selected:
				self._last_selected = selected[0][0]
			if self._last_selected is None:
				return
			self.model.load_distance(self._last_selected)
		else:
			self.model.load_clustering()
	
	def on_show_ids(self, *args):
		
		self.model.load_ids()
	
	def on_zoom(self, value):
		
		self.image_lst.set_thumbnail_size(value)

