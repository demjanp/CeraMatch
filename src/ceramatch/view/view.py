from deposit_gui import (DView, DNotification)

from ceramatch import __version__, __title__, res

from PySide6 import (QtWidgets, QtCore, QtGui)
import traceback
import os

class View(DView):
	
	APP_NAME = __title__
	VERSION = __version__
	
	def __init__(self, vcontrols, vgraph) -> None:
		
		DView.__init__(self)
		
		self._close_callback = None
		
		self.set_res_folder(os.path.dirname(res.__file__))
		
		central_widget = QtWidgets.QWidget(self)
		central_widget.setLayout(QtWidgets.QVBoxLayout())
		central_widget.layout().setContentsMargins(0, 0, 0, 0)
		central_widget.layout().setSpacing(0)
		self.setCentralWidget(central_widget)
		
		self._tool_window = QtWidgets.QMainWindow()
		self._tool_window.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
		central_widget.layout().addWidget(self._tool_window)
		
		central_frame = QtWidgets.QFrame()
		central_frame.setLayout(QtWidgets.QHBoxLayout())
		central_frame.layout().setContentsMargins(0, 0, 0, 0)
		central_widget.layout().addWidget(central_frame)
		
		self.dummy_view = QtWidgets.QGraphicsView()
		self.dummy_view.hide()
		
		central_frame.layout().addWidget(vcontrols)
		central_frame.layout().addWidget(vgraph)
		central_frame.layout().addWidget(self.dummy_view)
		
		self._notification = DNotification(vgraph)
		
		self.setWindowIcon(self.get_icon("cm_icon.svg"))
	
	def show_notification(self, text, delay = None):
		
		self._notification.show(text, delay)
	
	def hide_notification(self):
		
		self._notification.hide()
	
	# events
	
	def exception_event(self, typ, value, tb):
		
		error_title = "%s: %s" % (str(typ), str(value)[:512])
		self.show_notification('''
Application Error: %s
(see Help -> Log File for details)
		''' % (error_title),
			delay = 7000,
		)
		text = "Exception: %s\nTraceback: %s" % (
			error_title, 
			"".join(traceback.format_tb(tb)),
		)
		self.logging.append(text)
		print(text)
	
	# overriden QMainWindow methods
	
	def addToolBar(self, title):
		
		return self._tool_window.addToolBar(title)
	
	def addToolBarBreak(self):
		
		self._tool_window.addToolBarBreak()
	
	def closeEvent(self, event):
		
		if self._close_callback is not None:
			if not self._close_callback():
				event.ignore()
				return
		DView.closeEvent(self, event)
