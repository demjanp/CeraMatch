from lib.View import (View)

from PySide2 import (QtWidgets)
import sys

class CeraMatchMain(object):
	
	def __init__(self):
		
		app = QtWidgets.QApplication(sys.argv)
		self.view = View()
		self.view.show()
		app.exec_()
