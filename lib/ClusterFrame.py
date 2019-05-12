from lib.Slider import (Slider)

from PySide2 import (QtWidgets, QtCore, QtGui)

class ClusterFrame(QtWidgets.QFrame):
	
	def __init__(self, view):
		
		self.view = view
		self.model = view.model
		
		QtWidgets.QFrame.__init__(self, view)
		
		self.setLayout(QtWidgets.QVBoxLayout())
		self.layout().setContentsMargins(0, 0, 0, 0)
		
		self.clust_group = QtWidgets.QGroupBox("Clustering")
		self.clust_group.setLayout(QtWidgets.QVBoxLayout())
		self.clust_button = QtWidgets.QPushButton("Apply")
		self.clust_button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
		self.clust_button.clicked.connect(self.on_clust)
		max_clusters = len(self.model.sample_ids) // 2
		self.clust_slider = Slider(self.view, "cluster", 2, max_clusters)
		self.clust_slider.set_value(max_clusters)
		self.clust_group.layout().addWidget(self.clust_slider)
		self.clust_group.layout().addWidget(self.clust_button)
		self.layout().addWidget(self.clust_group)
		
		self.subclust_group = QtWidgets.QGroupBox("Sub-Clustering")
		self.subclust_group.setLayout(QtWidgets.QVBoxLayout())
		
		self.subclust_button_frame = QtWidgets.QFrame()
		self.subclust_button_frame.setContentsMargins(0, 0, 0, 0)
		self.subclust_button_frame.setLayout(QtWidgets.QHBoxLayout())
		
		self.subclust_apply = QtWidgets.QPushButton("Apply")
		self.subclust_apply.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
		self.subclust_apply.clicked.connect(self.on_subclust_apply)

		self.subclust_clear = QtWidgets.QPushButton("Clear")
		self.subclust_clear.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
		self.subclust_clear.clicked.connect(self.on_subclust_clear)
		
		self.subclust_button_frame.layout().addWidget(self.subclust_apply)
		self.subclust_button_frame.layout().addWidget(self.subclust_clear)
		
		self.subclust_slider = Slider(self.view, "subcluster", 2, 100)
		self.sublcust_cluster_label = QtWidgets.QLabel("Cluster:")
		self.sublcust_subcluster_label = QtWidgets.QLabel("Sub-Cluster:")
		self.subclust_group.layout().addWidget(self.sublcust_cluster_label)
		self.subclust_group.layout().addWidget(self.sublcust_subcluster_label)
		self.subclust_group.layout().addWidget(self.subclust_slider)
		self.subclust_group.layout().addWidget(self.subclust_button_frame)
		
		self.layout().addWidget(self.subclust_group)
	
	def update(self):
		
		enabled = False
		cluster = ""
		subcluster = ""
		selected = self.view.get_selected() # [Sample, ...]
		if selected:
			if selected[0].cluster is not None:
				cluster = selected[0].cluster
			if selected[0].subcluster is not None:
				subcluster = selected[0].subcluster
				if cluster in self.model.subcluster_numbers:
					self.subclust_slider.set_value(self.model.subcluster_numbers[cluster])
		self.sublcust_cluster_label.setText("Cluster: %s" % (cluster))
		self.sublcust_subcluster_label.setText("Sub-Cluster: %s" % (subcluster))
		
		if cluster:
			samples_n = len([sample for sample in self.model.samples if sample.cluster == cluster])
			self.subclust_slider.set_maximum(max(2, samples_n // 2))
		
		enabled = (cluster != "")
		self.subclust_apply.setEnabled(enabled)
		self.subclust_slider.setEnabled(enabled)
		
		self.subclust_clear.setEnabled(subcluster != "")
		
	def on_clust(self, *args):
		
		clusters_n = self.clust_slider.get_value()
		self.model.load_clustering(clusters_n)
	
	def on_subclust_apply(self, *args):
		
		clusters_n = self.subclust_slider.get_value()
		self.model.load_subclustering(clusters_n)
	
	def on_subclust_clear(self, *args):
		
		self.model.clear_subclustering()

