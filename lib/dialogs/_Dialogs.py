from deposit.DModule import (DModule)

from lib.dialogs.Connect import (Connect)
from lib.dialogs.Import import (Import)
from lib.dialogs.About import (About)

from PySide2 import (QtWidgets, QtCore, QtGui)

class Dialogs(DModule):
	
	def __init__(self, view):
		
		self.view = view
		self.model = view.model
		self.dialogs_open = []
		self.dialogs = dict([(cls.__name__, cls) for cls in [  # {name: Dialog, ...}
			Connect,
			Import,
			About,
		]])
		
		DModule.__init__(self)
		
	def open(self, dialog_name, *args, **kwargs):
		
		if dialog_name in self.dialogs:
			dialog = self.dialogs[dialog_name](self.model, self.view, *args)
			self.dialogs_open.append(dialog_name)
			dialog.show()
			return dialog
	
	def on_finished(self, code, dialog):
		
		dialog._closed = True
		
		self.dialogs_open.remove(dialog.__class__.__name__)
		
		if code == QtWidgets.QDialog.Accepted:
			dialog.process()
		elif code == QtWidgets.QDialog.Rejected:
			dialog.cancel()

