from deposit_gui import AbstractSubcontroller
from deposit_gui.view.vquerytoolbar import VQueryToolbar

from ceramatch.view.view import View

from PySide6 import (QtWidgets, QtCore, QtGui)
from pathlib import Path
import json
import os

class CView(AbstractSubcontroller):
	
	def __init__(self, cmain, cnavigator, cgraph) -> None:
		
		AbstractSubcontroller.__init__(self, cmain)
		
		self._view = View(cnavigator._view, cgraph._view)
		self._vquerytoolbar = None
		
		cgraph._view.set_button_zoom_reset(icon = self._view.get_icon("zoom_reset.svg"))
		cgraph._view.show_search_box(icon = self._view.get_icon("search.svg"))
		
		self.progress = self._view.progress
		
		self._view._close_callback = self.cmain.on_close
	
	def init_query_toolbar(self):
		
		self._view.addToolBarBreak()
		self._vquerytoolbar = VQueryToolbar(self._view)
		self.update_query()
		self._vquerytoolbar.signal_entered.connect(self.on_query_entered)
	
	def show(self):
		
		self._view.show()
	
	
	# ---- Signal handling
	# ------------------------------------------------------------------------
	@QtCore.Slot(str)
	def on_query_entered(self, querystr):
		
		self.cmain.cmodel.set_query(querystr)
		self.cmain.cmodel.load_drawings()
		self.cmain.ccontrols.update()
	
	def on_loaded(self):
		
		self.update_query()
	
	
	# ---- get/set
	# ------------------------------------------------------------------------
	def get_default_folder(self):
		
		folder = self._view.get_recent_dir()
		if folder:
			return folder
		
		if self.cmain.cmodel.has_local_folder():
			return self.cmain.cmodel.get_folder()
		
		return str(Path.home())
	
	def get_save_path(self, caption, filter, filename = None):
		# returns path, format
		
		folder = self.get_default_folder()
		if filename is not None:
			folder = os.path.join(folder, filename)
		path, format = QtWidgets.QFileDialog.getSaveFileName(self._view, dir = folder, caption = caption, filter = filter)
		
		return path, format
	
	def get_load_path(self, caption, filter):
		
		path, format = QtWidgets.QFileDialog.getOpenFileName(self._view, dir = self.get_default_folder(), caption = caption, filter = filter)
		
		return path, format
	
	def get_logging_path(self):
		
		return self._view.logging.get_log_path()
	
	def get_existing_folder(self, caption):
		
		folder = QtWidgets.QFileDialog.getExistingDirectory(self._view, dir = self.get_default_folder(), caption = caption)
		
		return folder
	
	def get_recent_dir(self):
		
		return self._view.get_recent_dir()
	
	def set_registry(self, name, data):
		
		self._view.registry.set(name, json.dumps(data))
	
	def get_registry(self, name, default = None):
		
		data = self._view.registry.get(name)
		if not data:
			return default
		return json.loads(data)
	
	def set_title(self, title):
		
		self._view.set_title(title)
	
	def update_query(self):
		
		querystr = "SELECT [%s]" % (self.cmain.cmodel.get_primary_class())
		self._vquerytoolbar.set_query_text(querystr)
		self.cmain.cmodel.set_query(querystr)
	
	def get_query_text(self):
		
		return self._vquerytoolbar.get_query_text()
	
	def set_recent_dir(self, path):
		
		if os.path.isfile(path):
			path = os.path.dirname(path)
		if not os.path.isdir(path):
			return
		self._view.set_recent_dir(path)
	
	def set_status_message(self, text):
		
		self._view.statusbar.message(text)
	
	def log_message(self, text):
		
		self._view.logging.append(text)
	
	def show_notification(self, text, delay = None):
		
		self._view.show_notification(text, delay)
	
	def hide_notification(self):
		
		self._view.hide_notification()
	
	def show_information(self, caption, text):
		
		QtWidgets.QMessageBox.information(self._view, caption, text)
	
	def show_warning(self, caption, text):
		
		QtWidgets.QMessageBox.warning(self._view, caption, text)
	
	def show_question(self, caption, text):
		
		reply = QtWidgets.QMessageBox.question(self._view, caption, text)
		
		return reply == QtWidgets.QMessageBox.Yes
	
	def show_input_dialog(self, caption, text, value = "", **kwargs):
		
		return self._view.show_input_dialog(caption, text, value, **kwargs)
	
	def set_dummy_graph(self, state):
		
		self._view.dummy_view.setVisible(state)
	
	def close(self):
		
		self._view.close()

