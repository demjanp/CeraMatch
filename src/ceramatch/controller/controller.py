from ceramatch.controller.cmodel import CModel
from ceramatch.controller.cview import CView
from ceramatch.controller.cgraph import CGraph
from ceramatch.controller.ccontrols import CControls
from ceramatch.controller.cdialogs import CDialogs
from ceramatch.controller.cactions import CActions
from ceramatch.controller.chistory import CHistory

from ceramatch import __version__

from deposit_gui.controller.controller import Controller as DepositController

from PySide6 import (QtWidgets, QtCore, QtGui)
import json
import sys
import os

class Controller(QtCore.QObject):
	
	def __init__(self):
		
		QtCore.QObject.__init__(self)
		
		self._deposit_controller = None
		
		self.history = CHistory(self)
		self.cmodel = CModel(self)
		self.ccontrols = CControls(self)
		self.cgraph = CGraph(self)
		self.cview = CView(self, self.ccontrols, self.cgraph)
		self.cdialogs = CDialogs(self, self.cview)
		self.cactions = CActions(self, self.cview)
		
		self.cmodel.set_progress(self.cview.progress)
		self.cmodel.load_descriptors()
		
		self.cview.log_message("CeraMatch started")
		
		self.cview.show()
		
		self.cview.init_query_toolbar()
		
		self.ccontrols.update()
		self.cmodel.update_model_info()
		self.cdialogs.open("Connect")
	
	
	# ---- Signal handling
	# ------------------------------------------------------------------------
	def on_close(self):
		
		if not self.check_save():
			return False
		
		if self._deposit_controller is not None:
			self._deposit_controller.close()
		return True
	
	
	# ---- get/set
	# ------------------------------------------------------------------------
	def clear(self):
		
		self.ccontrols.clear()
		self.cgraph.clear()
		self.history.clear()
	
	def check_save(self):
		
		if not self.cmodel.is_saved():
			reply = QtWidgets.QMessageBox.question(self.cview._view, 
				"Exit", "Save changes to database?",
				QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel
			)
			if reply == QtWidgets.QMessageBox.Yes:
				self.cactions.on_Save(True)
			elif reply == QtWidgets.QMessageBox.No:
				return True
			else:
				return False
		
		return True
	
	def open_deposit(self, querystr = None):
		
		if self._deposit_controller is None:
			self._deposit_controller = DepositController(
				self.cmodel._model._store
			)
		self._deposit_controller.cview.show()
		if querystr is not None:
			self._deposit_controller.cmdiarea.add_query(querystr)
	
	def open_folder(self, path):
		
		if sys.platform in ["linux", "linux2", "darwin"]:
			return # TODO
		if sys.platform.startswith("win"):
			if os.path.isdir(path):
				os.startfile(path)
	
	def open_in_external(self, path):
		
		if sys.platform in ["linux", "linux2", "darwin"]:
			return # TODO		
		if sys.platform.startswith("win"):
			if os.path.isfile(path):
				os.startfile(path)

