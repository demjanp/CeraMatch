#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import lib
from lib import (__version__, __date__)

from PySide2 import (QtWidgets, QtCore, QtGui)
import os

class About(QtWidgets.QDialog):

	def __init__(self, model, view, *args):
		
		self.model = model
		self.view = view
		self.buttonBox = None

		QtWidgets.QDialog.__init__(self, self.view)
		
		self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
		self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
		
		self.set_up(*args)

		self.setWindowTitle("About CeraMatch")
		
		self.buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok, QtCore.Qt.Horizontal)
		self.buttonBox.accepted.connect(self.accept)
		QtWidgets.QDialog.layout(self).addWidget(self.buttonBox)
		self.adjustSize()
	
	def set_up(self, *args):

		self.setMinimumWidth(400)
		self.setModal(True)
		self.setLayout(QtWidgets.QVBoxLayout())
		
		self.content = QtWidgets.QFrame()
		self.content.setLayout(QtWidgets.QHBoxLayout())
		self.layout().addWidget(self.content)
		
		self.logo = QtWidgets.QLabel()
		self.logo.setPixmap(QtGui.QPixmap("res/cm_logo.svg"))
		path = os.path.join(os.path.dirname(lib.__file__), "..", "THIRDPARTY.TXT").replace("\\", "/")
		self.label = QtWidgets.QLabel('''
<h2>CeraMatch</h2>
<h4>Shape matching and clustering of ceramic shapes</h4>
<p>Version %s (%s)</p>
<p>Copyright © <a href="mailto:peter.demjan@gmail.com">Peter Demján</a> 2019 - %s</p>
<p>&nbsp;</p>
<p>This application uses the Graph-based data storage <a href="https://github.com/demjanp/deposit">Deposit</a></p>
<p>&nbsp;</p>
<p>Licensed under the <a href="https://www.gnu.org/licenses/gpl-3.0.en.html">GNU General Public License</a></p>
<p><a href="https://github.com/demjanp/CeraMatch">Home page</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="%s">Third party libraries</a></p>
		''' % (__version__, __date__, __date__.split(".")[-1], path))
		self.label.setOpenExternalLinks(True)
		self.content.layout().addWidget(self.logo)
		self.content.layout().addWidget(self.label)

