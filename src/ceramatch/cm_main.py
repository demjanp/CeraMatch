from ceramatch.controller.controller import Controller
from deposit.utils.fnc_files import (clear_temp_dir)

from PySide6 import (QtWidgets)
import sys

class CMMain(object):
	
	def __init__(self):
		
		app = QtWidgets.QApplication(sys.argv)
#		app.setStyle("Fusion")
		
		self.controller = Controller()
		
		app.exec_()
	
	def __del__(self):
		
		clear_temp_dir(appdir = "ceramatch")
