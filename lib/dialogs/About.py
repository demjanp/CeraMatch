#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from lib.dialogs._Dialog import (Dialog)
from lib import (__version__, __date__, __file__)

from PySide2 import (QtWidgets, QtCore, QtGui)
import os

class About(Dialog):
	
	def title(self):
		
		return "About CeraMatch"
	
	def set_up(self):
		
		self.setMinimumWidth(400)
		self.setModal(True)
		self.setLayout(QtWidgets.QVBoxLayout())
		
		self.content = QtWidgets.QFrame()
		self.content.setLayout(QtWidgets.QHBoxLayout())
		self.layout().addWidget(self.content)
		
		self.logo = QtWidgets.QLabel()
		self.logo.setPixmap(QtGui.QPixmap("res/cm_logo.svg"))
		path = os.path.join(os.path.dirname(__file__), "..", "THIRDPARTY.TXT").replace("\\", "/")
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
		
		self.logobox = QtWidgets.QFrame()
		self.logobox.setLayout(QtWidgets.QVBoxLayout())
		self.logobox.layout().addWidget(self.logo)
		self.logobox.layout().addStretch()
		
		self.labelbox = QtWidgets.QFrame()
		self.labelbox.setLayout(QtWidgets.QVBoxLayout())
		self.labelbox.layout().addWidget(self.label)
		self.labelbox.layout().addStretch()
		
		self.content.layout().addWidget(self.logobox)
		self.content.layout().addWidget(self.labelbox)
		self.content.layout().addStretch()
	
	def button_box(self):
		
		return True, False
