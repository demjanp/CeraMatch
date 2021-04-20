from deposit.commander.dialogs.Connect import Connect as DConnect
from deposit.commander.dialogs.Connect import ClickableLogo

from PySide2 import (QtWidgets, QtCore, QtGui)

class Connect(DConnect):
	
	def __init__(self, model, view, *args):
		
		DConnect.__init__(self, model, view, *args)
		
		self.setParent(self.view, self.view.windowFlags())
	
	def logo(self):
		
		logo_frame = QtWidgets.QFrame()
		logo_frame.setLayout(QtWidgets.QVBoxLayout())
		logo_frame.layout().setContentsMargins(10, 10, 10, 10)
		logo_frame.layout().addWidget(ClickableLogo("res/cm_logo.svg", "https://github.com/demjanp/CeraMatch", alignment = QtCore.Qt.AlignCenter))
		logo_frame.layout().addStretch()
		
		return logo_frame
	
	def on_connect(self, identifier, connstr, local_folder = None, created = False):
		
		self.view.progress.show("Loading...")
		DConnect.on_connect(self, identifier, connstr, local_folder, created, no_import = True)
		self.view.progress.reset()