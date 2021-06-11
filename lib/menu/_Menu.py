from deposit import Broadcasts
from deposit.DModule import (DModule)

from lib.toolbar.Connect import Connect
from lib.toolbar.Save import Save
from lib.toolbar.Deposit import Deposit
from lib.toolbar.Undo import Undo
from lib.toolbar.Redo import Redo
from lib.toolbar.ImportClustering import ImportClustering
from lib.toolbar.ExportClustering import ExportClustering
from lib.toolbar.ExportDendrogram import ExportDendrogram
from lib.toolbar.ExportCatalog import ExportCatalog
from lib.toolbar._Toolbar import Separator

from lib.menu.SaveAsFile import SaveAsFile
from lib.menu.About import About
from lib.menu.ClearRecent import ClearRecent

from PySide2 import (QtWidgets, QtGui)
import json
import os

MENUS = {
	"Data": [Connect, Save, SaveAsFile, Deposit, Separator, ImportClustering, ExportClustering, Separator, ClearRecent,],
	"Edit": [Undo, Redo,],
	"Export PDF": [ExportDendrogram, ExportCatalog,],
	"Help": [About,],
}

RECENT_IN = "Data"

class Menu(DModule):
	
	def __init__(self, view):
		
		self.view = view
		self.model = view.model
		self.menubar = None
		self.actions = {}  # {name: QAction, ...}
		self.tools = {}  # {name: Tool, ...}
		self.recent_menu = None
		
		DModule.__init__(self)
		
		self.set_up()
	
	def set_up(self):
		
		self.menubar = self.view.menuBar()
		
		menus = {} # {name: QMenu, ...}
		for menu_name in MENUS:
			menus[menu_name] = self.menubar.addMenu(menu_name)
			for ToolClass in MENUS[menu_name]:
				if ToolClass == Separator:
					menus[menu_name].addSeparator()
				else:
					name = ToolClass.__name__
					self.tools[name] = ToolClass(self.model, self.view)
					self.actions[name] = QtWidgets.QAction(self.tools[name].name(), self.view)
					self.actions[name].setData(name)
					menus[menu_name].addAction(self.actions[name])
		self.recent_menu = menus[RECENT_IN]
		self.recent_menu.addSeparator()
		
		self.menubar.triggered.connect(self.on_triggered)
		
		self.connect_broadcast(Broadcasts.VIEW_ACTION, self.update_tools)
		self.connect_broadcast(Broadcasts.STORE_LOADED, self.update_tools)
		self.connect_broadcast(Broadcasts.STORE_LOCAL_FOLDER_CHANGED, self.update_tools)
		self.connect_broadcast(Broadcasts.STORE_DATA_SOURCE_CHANGED, self.update_tools)
		self.connect_broadcast(Broadcasts.STORE_DATA_CHANGED, self.update_tools)
		self.connect_broadcast(Broadcasts.STORE_SAVED, self.update_tools)
		
		self.update_tools()
	
	def update_tools(self, *args):
		
		for name in self.tools:
			tool_name = self.tools[name].name()
			icon = self.tools[name].icon()
			help = self.tools[name].help()
			checkable = self.tools[name].checkable()
			shortcut = self.tools[name].shortcut()
			self.actions[name].setText(tool_name)
			if icon:
				icon = self.view.get_icon(icon)
				self.actions[name].setIcon(icon)
			if shortcut:
				self.actions[name].setShortcut(QtGui.QKeySequence(shortcut))
			self.actions[name].setCheckable(checkable)
			self.actions[name].setToolTip(help)
			self.actions[name].setChecked(self.tools[name].checked())
			self.actions[name].setEnabled(self.tools[name].enabled())
	
	def load_recent(self):
		
		rows = self.view.registry.get("recent")
		if rows == "":
			return
		rows = json.loads(rows)
		for row in rows:
			if len(row) == 1:
				self.add_recent_url(row[0])
			elif len(row) == 2:
				self.add_recent_db(*row)
		
	def save_recent(self):
		
		rows = []
		for action in self.recent_menu.actions():
			if action.parent() is None:
				continue
			data = action.data()
			if isinstance(data, list):
				rows.append(data)
		self.view.registry.set("recent", json.dumps(rows))
		
	def get_recent(self):
		# return [[url], [identifier, connstr], ...]
		
		collect = []
		for action in self.recent_menu.actions():
			data = action.data()
			if isinstance(data, list):
				collect.append(data)
		return collect
	
	def clear_recent(self):
		
		for action in self.recent_menu.actions():
			if isinstance(action.data(), list):
				action.setParent(None)
		self.save_recent()
	
	def has_recent(self, data):
		
		for action in self.recent_menu.actions():
			if action.data() == data:
				return True
		return False
	
	def add_recent_url(self, url):
		
		if self.has_recent([url]):
			return
		action = QtWidgets.QAction(url, self.view)
		action.setData([url])
		self.recent_menu.addAction(action)
		self.save_recent()
	
	def add_recent_db(self, identifier, connstr):
		
		if (not identifier) or (not connstr):
			return
		if self.has_recent([identifier, connstr]):
			return
		name = "%s (%s)" % (identifier, os.path.split(connstr)[1])
		action = QtWidgets.QAction(name, self.view)
		action.setData([identifier, connstr])
		self.recent_menu.addAction(action)
		self.save_recent()
	
	def on_triggered(self, action):

		data = action.data()
		if isinstance(data, list):
			self.on_recent_triggered(data)
			return
		self.tools[str(data)].triggered(action.isChecked())
		self.update_tools()
	
	def on_recent_triggered(self, data):
		
		if len(data) == 1:
			url = data[0]
			self.model.load(url)
		elif len(data) == 2:
			identifier, connstr = data
			self.model.load(identifier, connstr)
	
