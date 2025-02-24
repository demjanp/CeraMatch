
from ceramatch.view.vcontrols_groups.controls.button import (Button)
from ceramatch.view.vcontrols_groups.controls.combo import (Combo)
from ceramatch.view.vcontrols_groups.controls.line_edit import (LineEdit)

from PySide6 import (QtWidgets, QtCore, QtGui)

class Clusters(QtWidgets.QGroupBox):
	
	signal_cluster = QtCore.Signal()
	signal_update_tree = QtCore.Signal()
	signal_add_cluster = QtCore.Signal()
	signal_rename_cluster = QtCore.Signal()
	signal_delete = QtCore.Signal()
	signal_n_clusters_changed = QtCore.Signal()
	
	def __init__(self):
		
		QtWidgets.QGroupBox.__init__(self, "Clustering")
		
		self.setStyleSheet("QGroupBox {font-weight: bold;}")
		self.setLayout(QtWidgets.QVBoxLayout())
		
		self.cluster_button = Button("Auto Cluster", self.on_cluster)
		self.n_clusters_combo = Combo(self.on_n_clusters_changed)
		self.limit_edit = LineEdit("0.68")
		self.n_samples_label = QtWidgets.QLabel("")
		self.n_clusters_label = QtWidgets.QLabel("")
		self.update_tree_button = Button("Update Tree", self.on_update_tree)
		self.add_cluster_button = Button("Make Cluster", self.on_add_cluster)
		self.rename_cluster_button = Button("Rename Cluster", self.on_rename_cluster)
		self.delete_button = Button("Clear Clusters", self.on_delete)
		
		form_frame = QtWidgets.QFrame()
		form_frame.setLayout(QtWidgets.QFormLayout())
		form_frame.layout().setContentsMargins(0, 0, 0, 0)
		form_frame.layout().addRow(
			QtWidgets.QLabel("N Clusters:"), self.n_clusters_combo
		)
		form_frame.layout().addRow(
			QtWidgets.QLabel("Dist. Limit:"), self.limit_edit
		)
		form_frame.layout().addRow(
			QtWidgets.QLabel("Samples:"), self.n_samples_label
		)
		form_frame.layout().addRow(
			QtWidgets.QLabel("Clusters Found:"), self.n_clusters_label
		)
		
		self.layout().addWidget(self.cluster_button)
		self.layout().addWidget(form_frame)
		self.layout().addWidget(self.update_tree_button)
		self.layout().addWidget(self.add_cluster_button)
		self.layout().addWidget(self.rename_cluster_button)
		self.layout().addWidget(self.delete_button)
	
	# ---- Signal handling
	# ------------------------------------------------------------------------
	@QtCore.Slot()
	def on_cluster(self):
		
		self.signal_cluster.emit()
	
	@QtCore.Slot()
	def on_update_tree(self):
		
		self.signal_update_tree.emit()
	
	@QtCore.Slot()
	def on_add_cluster(self):
		
		self.signal_add_cluster.emit()
	
	@QtCore.Slot()
	def on_rename_cluster(self):
		
		self.signal_rename_cluster.emit()
	
	@QtCore.Slot()
	def on_delete(self):
		
		self.signal_delete.emit()
	
	@QtCore.Slot()
	def on_n_clusters_changed(self):
		
		self.signal_n_clusters_changed.emit()
	
	
	# ---- get/set
	# ------------------------------------------------------------------------
	def get_limits(self):
		
		try:
			n_clusters = int(self.n_clusters_combo.get_value())
		except:
			n_clusters = None
		try:
			limit = float(self.limit_edit.text())
		except:
			limit = None
		return n_clusters, limit
	
	def set_clustering_enabled(self, state):
		
		self.cluster_button.setEnabled(state)
		self.n_clusters_combo.setEnabled(state)
		self.limit_edit.setEnabled(state)
	
	def set_limit_enabled(self, state):
		
		self.limit_edit.setEnabled(state)
	
	def set_update_tree_enabled(self, state):
		
		self.update_tree_button.setEnabled(state)
	
	def set_add_cluster_enabled(self, state):
		
		self.add_cluster_button.setEnabled(state)
	
	def set_rename_cluster_enabled(self, state):
		
		self.rename_cluster_button.setEnabled(state)
	
	def set_delete_enabled(self, state):
		
		self.delete_button.setEnabled(state)
	
	def set_n_clusters(self, values, n_clusters = None):
		
		self.n_clusters_combo.set_values(values, n_clusters)
	
	def set_n_samples(self, n_samples):
		
		if n_samples is None:
			n_samples = ""
		self.n_samples_label.setText(str(n_samples))
	
	def set_n_found(self, n_found):
		
		if n_found is None:
			n_found = ""
		self.n_clusters_label.setText(str(n_found))

