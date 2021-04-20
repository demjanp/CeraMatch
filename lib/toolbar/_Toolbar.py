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

from PySide2 import (QtWidgets, QtCore, QtGui)

class Separator():
	
	pass

class Toolbar(DModule):
	
	TOOLS = {
		"Data": [
			Connect, Save, Deposit, Separator,
			Undo, Redo, Separator,
			ImportClustering, ExportClustering, Separator,
			ExportDendrogram, ExportCatalog,
		],
	}
	
	def __init__(self, view):
		
		self.view = view
		self.model = view.model
		self.toolbars = {} # {name: QToolBar, ...}
		self.actions = {}  # {name: QAction, ...}
		self.tools = {}  # {name: Tool, ...}
		
		DModule.__init__(self)
		
		self.set_up()
	
	def set_up(self):
		
		for toolbar_name in self.TOOLS:
			self.toolbars[toolbar_name] = self.view.addToolBar(toolbar_name)
			self.toolbars[toolbar_name].setIconSize(QtCore.QSize(36,36))
			for ToolClass in self.TOOLS[toolbar_name]:
				if ToolClass == Separator:
					self.toolbars[toolbar_name].addSeparator()
				else:
					name = ToolClass.__name__
					self.tools[name] = ToolClass(self.model, self.view)
					self.actions[name] = QtWidgets.QAction(self.tools[name].name(), self.view)
					self.actions[name].setData(name)
					self.toolbars[toolbar_name].addAction(self.actions[name])
			self.toolbars[toolbar_name].actionTriggered.connect(self.on_triggered)
		
		self.connect_broadcast(Broadcasts.VIEW_ACTION, self.update_tools)
		self.connect_broadcast(Broadcasts.STORE_LOADED, self.update_tools)
		self.connect_broadcast(Broadcasts.STORE_LOCAL_FOLDER_CHANGED, self.update_tools)
		self.connect_broadcast(Broadcasts.STORE_DATA_SOURCE_CHANGED, self.update_tools)
		self.connect_broadcast(Broadcasts.STORE_DATA_CHANGED, self.update_tools)
		self.connect_broadcast(Broadcasts.STORE_SAVED, self.update_tools)
		
		self.update_tools()
	
	def update_tools(self, *args):
		
		for name in self.tools:
			self.actions[name].setText(self.tools[name].name())
			icon = self.tools[name].icon()
			if icon:
				icon = self.view.get_icon(icon)
				self.actions[name].setIcon(icon)
			shortcut = self.tools[name].shortcut()
			if shortcut:
				self.actions[name].setShortcut(QtGui.QKeySequence(shortcut))
				self.actions[name].setShortcutContext(QtCore.Qt.WindowShortcut)
			self.actions[name].setCheckable(self.tools[name].checkable())
			self.actions[name].setToolTip(self.tools[name].help())
			self.actions[name].setChecked(self.tools[name].checked())
			self.actions[name].setEnabled(self.tools[name].enabled())
	
	def on_triggered(self, action):

		self.tools[str(action.data())].triggered(action.isChecked())
		self.update_tools()
	