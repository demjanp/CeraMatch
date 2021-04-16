from deposit.commander.menu._MRUDMenu import (MRUDMenu)
from deposit import Broadcasts

from PySide2 import (QtWidgets, QtCore, QtGui)
import json
import os

class Menu(MRUDMenu):
	
	def __init__(self, view):
		
		self.menubar = None
		
		MRUDMenu.__init__(self, view.model, view)
	
	def set_up(self):
		
		self.menubar = self.view.menuBar()
		
		self.menu_data = self.menubar.addMenu("Data")
		self.action_connect = QtWidgets.QAction(QtGui.QIcon("res\connect.svg"), "Connect Data Source", self.view)
		self.action_connect.setCheckable(True)
		self.action_connect.triggered.connect(self.on_connect)
		self.menu_data.addAction(self.action_connect)
		self.action_save = QtWidgets.QAction(QtGui.QIcon("res\save.svg"), "Save", self.view)
		self.action_save.triggered.connect(self.on_save)
		self.menu_data.addAction(self.action_save)
		self.action_save_pdf = QtWidgets.QAction(QtGui.QIcon("res\capture_pdf.svg"), "Save as PDF", self.view)
		self.action_save_pdf.triggered.connect(self.on_save_pdf)
		self.menu_data.addAction(self.action_save_pdf)
		self.action_deposit = QtWidgets.QAction(QtGui.QIcon("res\dep_cube.svg"), "Deposit", self.view)
		self.action_deposit.triggered.connect(self.on_deposit)
		self.menu_data.addAction(self.action_deposit)
		self.menu_data.addSeparator()
		self.action_clear_recent = QtWidgets.QAction("Clear Recent", self.view)
		self.menu_data.addAction(self.action_clear_recent)
		self.menu_data.addSeparator()
		
		menu = self.menubar.addMenu("Help")
		self.action_about = QtWidgets.QAction("About", self.view)
		self.action_about.triggered.connect(self.on_about)
		menu.addAction(self.action_about)
		
		self.connect_broadcast(Broadcasts.VIEW_ACTION, self.on_update)
		self.connect_broadcast(Broadcasts.STORE_LOADED, self.on_update)
		self.connect_broadcast(Broadcasts.STORE_LOCAL_FOLDER_CHANGED, self.on_update)
		self.connect_broadcast(Broadcasts.STORE_DATA_SOURCE_CHANGED, self.on_update)
		self.connect_broadcast(Broadcasts.STORE_DATA_CHANGED, self.on_update)
		self.connect_broadcast(Broadcasts.STORE_SAVED, self.on_update)
		
		self.recent_menu = self.menu_data
		
		self.update()
		
	def update(self):
		
		if not self.model.is_connected():
			self.action_save.setEnabled(False)
			self.action_save_pdf.setEnabled(False)
		else:
			self.action_save.setEnabled(True)
			self.action_save_pdf.setEnabled(True)
	
	def on_update(self, *args):
		
		self.update()
	
	def on_connect(self, *args):
		
		self.view.toolbar.on_connect()
	
	def on_save(self, *args):
		
		self.view.toolbar.on_save()
	
	def on_save_pdf(self, *args):
		
		self.view.toolbar.on_save_pdf()
	
	def on_deposit(self, *args):
		
		self.view.toolbar.on_deposit()
	
	def on_about(self, *args):
		
		self.view.dialogs.open("About")
	