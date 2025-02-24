#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from ceramatch import __version__, __date__
import ceramatch

from PySide6 import (QtWidgets, QtCore, QtGui)
import os

class DialogAbout(QtWidgets.QFrame):
	
	def __init__(self):
		
		QtWidgets.QFrame.__init__(self)
		
		self.setMinimumWidth(400)
		self.setLayout(QtWidgets.QVBoxLayout())
		
		content = QtWidgets.QFrame()
		content.setLayout(QtWidgets.QHBoxLayout())
		self.layout().addWidget(content)
		
		logo = QtWidgets.QLabel()
		logo.setPixmap(QtGui.QPixmap(os.path.join(os.path.dirname(ceramatch.__file__), "res/cm_logo.svg")))
		path_third = os.path.join(os.path.dirname(ceramatch.__file__), "THIRDPARTY.TXT").replace("\\", "/")
		label = QtWidgets.QLabel('''
<h2>CeraMatch</h2>
<h4>Visual shape-matching and classification of ceramics</h4>
<p>Version %s (%s)</p>
<p>Copyright © <a href="mailto:peter.demjan@gmail.com">Peter Demján</a> 2019 - %s</p>
<p>&nbsp;</p>
<p>This application uses the Graph-based data storage <a href="https://github.com/demjanp/deposit">Deposit</a></p>
<p>&nbsp;</p>
<p>Licensed under the <a href="https://www.gnu.org/licenses/gpl-3.0.en.html">GNU General Public License</a></p>
<p><a href="https://github.com/demjanp/CeraMatch">Home page</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="%s">Third party libraries</a></p>
		''' % (__version__, __date__, __date__.split(".")[-1], path_third))
		label.setOpenExternalLinks(True)
		content.layout().addWidget(logo)
		content.layout().addWidget(label)
