
from lib.Button import (Button)
from lib.Combo import (Combo)

from deposit.DModule import (DModule)
from deposit import Broadcasts

from PySide2 import (QtWidgets, QtCore, QtGui)

class ClusterGroup(DModule, QtWidgets.QGroupBox):
	
	cluster = QtCore.Signal()
	
	def __init__(self, view):
		
		self.view = view
		self.model = view.model
		
		DModule.__init__(self)
		QtWidgets.QGroupBox.__init__(self, "Clustering")
		
		self.setLayout(QtWidgets.QVBoxLayout())
		
		self.cluster_button = Button("Cluster", self.on_cluster)
		self.n_clusters_combo = Combo(self.on_n_clusters_changed)
		self.limit_edit = QtWidgets.QLineEdit("0.68")
		self.limit_edit.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
		self.n_samples_label = QtWidgets.QLabel("")
		self.n_clusters_label = QtWidgets.QLabel("")
		
		form_frame = QtWidgets.QFrame()
		form_frame.setLayout(QtWidgets.QFormLayout())
		form_frame.layout().setContentsMargins(0, 0, 0, 0)
		form_frame.layout().addRow(QtWidgets.QLabel("N Clusters:"), self.n_clusters_combo)
		form_frame.layout().addRow(QtWidgets.QLabel("Dist. Limit:"), self.limit_edit)
		form_frame.layout().addRow(QtWidgets.QLabel("Samples:"), self.n_samples_label)
		form_frame.layout().addRow(QtWidgets.QLabel("Clusters Found:"), self.n_clusters_label)
		
		self.layout().addWidget(self.cluster_button)
		self.layout().addWidget(form_frame)
		
		self.update()
		
		self.connect_broadcast(Broadcasts.STORE_LOADED, self.on_data_source_changed)
		self.connect_broadcast(Broadcasts.STORE_DATA_SOURCE_CHANGED, self.on_data_source_changed)
		self.connect_broadcast(Broadcasts.STORE_DATA_CHANGED, self.on_data_changed)
	
	def get_limits(self):
		
		try:
			n_clusters = int(self.n_clusters_combo.get_value())
		except:
			n_clusters = None
		try:
			limit = float(self.limit_edit.text())
		except:
			limit = 0.68
		return n_clusters, limit
	
	def update(self):
		
		self.cluster_button.setEnabled(self.model.has_distance())
		self.n_clusters_combo.setEnabled(self.model.has_distance())
		self.limit_edit.setEnabled(self.model.has_distance() and (self.n_clusters_combo.get_value() == "By limit"))
		self.n_samples_label.setText(str(len(self.model.sample_ids)))
	
	def update_n_clusters(self):
		
		values = ["By limit"]
		if self.model.has_distance():
			values += list(range(2, len(self.model.sample_ids) + 1))
		self.n_clusters_combo.set_values(values)
		self.update()
	
	def update_clusters_found(self, n_clusters):
		
		if str(n_clusters).isnumeric():
			self.n_clusters_label.setText(str(n_clusters))
		else:
			self.n_clusters_label.setText("")
	
	@QtCore.Slot()
	def on_cluster(self):
		
		self.cluster.emit()
	
	@QtCore.Slot()
	def on_n_clusters_changed(self):
		
		self.update()
	
	def on_data_source_changed(self, *args):
		
		self.update_n_clusters()
		self.update_clusters_found(None)
		self.update()
	
	def on_data_changed(self, *args):
		
		self.update()
