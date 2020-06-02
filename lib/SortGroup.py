from lib.Button import (Button)

from PySide2 import (QtWidgets, QtCore, QtGui)

class SortGroup(QtWidgets.QGroupBox):
	
	def __init__(self, view):
		
		self.view = view
		self.model = view.model
		
		QtWidgets.QGroupBox.__init__(self, "Sort")
		
		self.setLayout(QtWidgets.QVBoxLayout())
		
		self.samples_button = Button("Sort by Sample IDs", self.view.on_samples)
		self.samples_button.setEnabled(False)
		self.distance_max_button = Button("Sort by Max Distance", self.view.on_distance_max)
		self.distance_max_button.setEnabled(False)
		self.distance_min_button = Button("Sort by Min Distance", self.view.on_distance_min)
		self.distance_min_button.setEnabled(False)
		self.cluster_button = Button("Sort by Clustering", self.view.on_cluster)
		self.cluster_button.setEnabled(False)
		
		self.layout().addWidget(self.samples_button)
		self.layout().addWidget(self.distance_max_button)
		self.layout().addWidget(self.distance_min_button)
		self.layout().addWidget(self.cluster_button)
	
	def update(self):
		
		selected = self.view.get_selected()
		
		self.samples_button.setEnabled(self.model.is_connected())
		self.distance_max_button.setEnabled(self.model.has_distances())
		self.distance_min_button.setEnabled(self.model.has_distances() and ((len(selected) > 0) or ((len(self.model.samples) > 0) and isinstance(self.model.samples[0].value, float))))
		self.cluster_button.setEnabled(self.model.has_clusters())

