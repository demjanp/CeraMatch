from lib.Button import (Button)

from PySide2 import (QtWidgets, QtCore, QtGui)

class ClusterGroup(QtWidgets.QGroupBox):
	
	def __init__(self, view):
		
		self.view = view
		self.model = view.model
		self.sliders = {}
		
		QtWidgets.QGroupBox.__init__(self, "Cluster")
		
		self.setLayout(QtWidgets.QVBoxLayout())
		
		self.split_button = Button("Split", self.view.on_split_cluster)
		self.join_button = Button("Join", self.view.on_join_cluster)
		self.manual_button = Button("Create Manual", self.view.on_manual_cluster)
		self.split_all_button = Button("Split All", self.view.on_split_all_clusters)
		self.outlier_button = Button("Set Outlier", self.view.on_set_outlier, icon = "reject.svg")
		self.central_button = Button("Set Central", self.view.on_set_central, icon = "accept.svg")
		
		self.layout().addWidget(self.split_button)
		self.layout().addWidget(self.join_button)
		self.layout().addWidget(self.manual_button)
		self.layout().addWidget(self.split_all_button)
		self.layout().addWidget(self.outlier_button)
		self.layout().addWidget(self.central_button)
	
	def update(self):
		
		selected = self.view.get_selected()
		
		has_cluster = ((len(selected) > 0) and (selected[0].has_cluster()))
		
		self.split_button.setEnabled(len(selected) > 0)
		
		self.join_button.setEnabled(has_cluster)
		
		self.manual_button.setEnabled(len(selected) > 1)
		
		self.outlier_button.setEnabled(has_cluster)
		
		self.central_button.setEnabled(has_cluster)

