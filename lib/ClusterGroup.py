from lib.Button import (Button)

from PySide2 import (QtWidgets, QtCore, QtGui)

class ClusterGroup(QtWidgets.QGroupBox):
	
	def __init__(self, view):
		
		self.view = view
		self.model = view.model
		self.sliders = {}
		
		QtWidgets.QGroupBox.__init__(self, "Cluster")
		
		self.setLayout(QtWidgets.QVBoxLayout())
		
		self.autoclust_button = Button("Auto Cluster", self.view.on_auto_cluster)
		self.split_button = Button("Split", self.view.on_split_cluster)
		self.join_parent_button = Button("Join to Parent", self.view.on_join_parent)
		self.join_children_button = Button("Join Children", self.view.on_join_children)
		self.split_at_selected_button = Button("Split At Selected", self.view.on_split_at_selected)
		self.manual_button = Button("Create Manual", self.view.on_manual_cluster)
		self.split_all_button = Button("Split All", self.view.on_split_all_clusters)
		self.outlier_button = Button("Set Outlier", self.view.on_set_outlier, icon = "reject.svg")
		self.central_button = Button("Set Central", self.view.on_set_central, icon = "accept.svg")
		self.clear_button = Button("Clear All", self.view.on_clear_clusters)
		
		self.layout().addWidget(self.autoclust_button)
		self.layout().addWidget(self.split_button)
		self.layout().addWidget(self.join_parent_button)
		self.layout().addWidget(self.join_children_button)
		self.layout().addWidget(self.split_at_selected_button)
		self.layout().addWidget(self.manual_button)
		self.layout().addWidget(self.split_all_button)
		self.layout().addWidget(self.outlier_button)
		self.layout().addWidget(self.central_button)
		self.layout().addWidget(self.clear_button)
	
	def update(self):
		
		selected = self.view.get_selected()
		
		has_cluster = ((len(selected) > 0) and (selected[0].has_cluster()))
		
		self.split_button.setEnabled(len(selected) > 0)
		
		self.split_at_selected_button.setEnabled(len(selected) == 1)
		
		self.join_parent_button.setEnabled(has_cluster)
		self.join_children_button.setEnabled(has_cluster and not self.view.image_view.is_list())
		
		self.manual_button.setEnabled(len(selected) > 0)
		
		self.outlier_button.setEnabled(has_cluster)
		
		self.central_button.setEnabled(has_cluster)
		
		self.clear_button.setEnabled(self.model.has_clusters())

