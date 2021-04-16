from deposit.commander.ViewChild import (ViewChild)
from deposit import Broadcasts

from PySide2 import (QtWidgets, QtCore, QtGui)
import os

class ToolBar(ViewChild):
	
	def __init__(self, view):
		
		ViewChild.__init__(self, view.model, view)
		
		self.last_dir = ""
		
		self.toolbar = self.view.addToolBar("ToolBar")
		self.toolbar.setIconSize(QtCore.QSize(36,36))
		
		self.action_connect = QtWidgets.QAction(QtGui.QIcon("res\connect.svg"), "Connect Data Source", self.view)
		self.action_connect.triggered.connect(self.on_connect)
		self.toolbar.addAction(self.action_connect)
		
		self.action_save = QtWidgets.QAction(QtGui.QIcon("res\save.svg"), "Save", self.view)
		self.action_save.triggered.connect(self.on_save)
		self.toolbar.addAction(self.action_save)
		
		self.action_deposit = QtWidgets.QAction(QtGui.QIcon("res\dep_cube.svg"), "Open Database in Deposit", self.view)
		self.action_deposit.triggered.connect(self.on_deposit)
		self.toolbar.addAction(self.action_deposit)
		
		self.toolbar.addSeparator()
		
		self.action_save_pdf = QtWidgets.QAction(QtGui.QIcon("res\capture_pdf.svg"), "Save as PDF", self.view)
		self.action_save_pdf.triggered.connect(self.on_save_pdf)
		self.toolbar.addAction(self.action_save_pdf)
		
		self.toolbar.addSeparator()
		
		self.action_undo = QtWidgets.QAction(QtGui.QIcon("res\\undo.svg"), "Undo", self.view)
		self.action_undo.triggered.connect(self.on_undo)
		self.toolbar.addAction(self.action_undo)
		
		self.connect_broadcast(Broadcasts.VIEW_ACTION, self.on_update)
		self.connect_broadcast(Broadcasts.STORE_LOADED, self.on_update)
		self.connect_broadcast(Broadcasts.STORE_LOCAL_FOLDER_CHANGED, self.on_update)
		self.connect_broadcast(Broadcasts.STORE_DATA_SOURCE_CHANGED, self.on_update)
		self.connect_broadcast(Broadcasts.STORE_DATA_CHANGED, self.on_update)
		self.connect_broadcast(Broadcasts.STORE_SAVED, self.on_update)
		
		self.update()
		
	def update(self):
		
		if not self.model.is_connected():
			self.action_save.setEnabled(False)
			self.action_save_pdf.setEnabled(False)
		else:
			self.action_save.setEnabled(True)
			self.action_save_pdf.setEnabled(True)
		
		self.action_undo.setEnabled(self.model.has_history())
	
	def on_update(self, *args):
		
		self.update()
	
	def on_connect(self, *args):
		
		self.view.dialogs.open("Connect")
	
	def on_save(self, *args):
		
		self.view.save()
	
	def on_deposit(self, *args):
		
		self.model.launch_deposit()
	
	def on_save_pdf(self, *args):
		
		path, _ = QtWidgets.QFileDialog.getSaveFileName(None, caption = "Save Clustering as PDF", filter = "(*.pdf)")
		if path:
			self.last_dir = os.path.split(path)[0]
			self.model.save_clusters_pdf(path)
	
	def on_undo(self, *args):
		
		self.model.undo_clustering()

