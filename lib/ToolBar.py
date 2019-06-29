from PySide2 import (QtWidgets, QtCore, QtGui)
import os

class ToolBar(object):
	
	def __init__(self, view):
		
		self.view = view
		self.model = view.model
		
		self.last_dir = ""
		
		self.toolbar = self.view.addToolBar("ToolBar")
		self.toolbar.setIconSize(QtCore.QSize(36,36))
		
		self.action_load = QtWidgets.QAction(QtGui.QIcon("res\open.svg"), "Load", self.view)
		self.action_load.triggered.connect(self.on_load)
		self.toolbar.addAction(self.action_load)

		self.action_save = QtWidgets.QAction(QtGui.QIcon("res\save.svg"), "Save", self.view)
		self.action_save.triggered.connect(self.on_save)
		self.toolbar.addAction(self.action_save)
		
		self.toolbar.addSeparator()
		
		self.action_save_pdf = QtWidgets.QAction(QtGui.QIcon("res\capture_pdf.svg"), "Save as PDF", self.view)
		self.action_save_pdf.triggered.connect(self.on_save_pdf)
		self.toolbar.addAction(self.action_save_pdf)
		
		self.toolbar.addSeparator()
		
		self.action_undo = QtWidgets.QAction(QtGui.QIcon("res\\undo.svg"), "Undo", self.view)
		self.action_undo.triggered.connect(self.on_undo)
		self.toolbar.addAction(self.action_undo)
	
	def update(self):
		
		self.action_undo.setEnabled(self.model.has_history())
	
	def on_load(self, *args):
		
		path, _ = QtWidgets.QFileDialog.getOpenFileName(self.view, caption = "Load Clustering", filter = "(*.json)", directory = self.last_dir)
		if path:
			self.last_dir = os.path.split(path)[0]
			self.model.load_clusters(path)
	
	def on_save(self, *args):
		
		path, _ = QtWidgets.QFileDialog.getSaveFileName(None, caption = "Save Clustering", filter = "(*.json)")
		if path:
			self.last_dir = os.path.split(path)[0]
			self.model.save_clusters(path)
	
	def on_save_pdf(self, *args):
		
		path, _ = QtWidgets.QFileDialog.getSaveFileName(None, caption = "Save Clustering as PDF", filter = "(*.pdf)")
		if path:
			self.last_dir = os.path.split(path)[0]
			self.model.save_clusters_pdf(path)
	
	def on_undo(self, *args):
		
		self.model.undo_clustering()
