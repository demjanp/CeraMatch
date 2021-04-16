from lib.Model import (Model)
from lib.dialogs._Dialogs import (Dialogs)
from lib.ToolBar import (ToolBar)
from lib.Menu import (Menu)
from lib.DescriptorGroup import (DescriptorGroup)
from lib.ClusterGroup import (ClusterGroup)
from lib.GraphView import (GraphView)
from lib.StatusBar import (StatusBar)
from lib.Button import (Button)


from deposit import Broadcasts
from deposit.commander.Registry import (Registry)
from deposit.DModule import (DModule)

from PySide2 import (QtWidgets, QtCore, QtGui)
import os

class View(DModule, QtWidgets.QMainWindow):
	
	def __init__(self):
		
		self.model = None
		
		DModule.__init__(self)
		QtWidgets.QMainWindow.__init__(self)
		
		self.model = Model(self)
		
		self.dialogs = Dialogs(self)
		self.registry = Registry("Deposit")
		self.descriptor_group = DescriptorGroup(self)
		self.cluster_group = ClusterGroup(self)
		self.graph_view = GraphView(self)
		self.menu = Menu(self)
		self.toolbar = ToolBar(self)
		self.statusbar = StatusBar(self)
		self.progress = None
		
		self.calculate_button = Button("Calculate Distances", self.on_calculate)
		self.calculate_button.setEnabled(False)
		
		self.setWindowIcon(QtGui.QIcon("res\cm_icon.svg"))
		self.setStyleSheet("QPushButton {padding: 5px; min-width: 100px;}")
		
		central_widget = QtWidgets.QWidget(self)
		central_widget.setLayout(QtWidgets.QHBoxLayout())
		central_widget.layout().setContentsMargins(0, 0, 0, 0)
		self.setCentralWidget(central_widget)
		
		control_frame = QtWidgets.QFrame(self)
		control_frame.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
		control_frame.setLayout(QtWidgets.QVBoxLayout())
		control_frame.layout().setContentsMargins(10, 10, 0, 10)
		
		graph_view_frame = QtWidgets.QFrame(self)
		graph_view_frame.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
		graph_view_frame.setLayout(QtWidgets.QVBoxLayout())
		graph_view_frame.layout().setContentsMargins(0, 0, 0, 0)
		
		central_widget.layout().addWidget(control_frame)
		central_widget.layout().addWidget(graph_view_frame)
		
		calculate_group = QtWidgets.QGroupBox("Calculate")
		calculate_group.setLayout(QtWidgets.QVBoxLayout())
		calculate_group.layout().addWidget(self.calculate_button)
		
		control_frame.layout().addWidget(self.descriptor_group)
		control_frame.layout().addWidget(calculate_group)
		control_frame.layout().addWidget(self.cluster_group)
		control_frame.layout().addStretch()
		
		graph_view_frame.layout().addWidget(self.graph_view)
		
		self.setStatusBar(self.statusbar)
		
		self.set_title()
#		self.setGeometry(100, 100, 1024, 768)
		self.setGeometry(500, 100, 1024, 768)  # DEBUG
		
		self.descriptor_group.load_data.connect(self.on_load_data)
		self.cluster_group.cluster.connect(self.on_cluster)
		
		self.connect_broadcast(Broadcasts.VIEW_ACTION, self.on_view_action)
		self.connect_broadcast(Broadcasts.STORE_LOADED, self.on_data_source_changed)
		self.connect_broadcast(Broadcasts.STORE_DATA_SOURCE_CHANGED, self.on_data_source_changed)
		self.connect_broadcast(Broadcasts.STORE_DATA_CHANGED, self.on_data_changed)
		self.connect_broadcast(Broadcasts.STORE_SAVED, self.on_saved)
		self.connect_broadcast(Broadcasts.STORE_SAVE_FAILED, self.on_save_failed)
		self.set_on_broadcast(self.on_broadcast)
		
		self.model.broadcast_timer.setSingleShot(True)
		self.model.broadcast_timer.timeout.connect(self.on_broadcast_timer)
		
		self.update()
		
		self.dialogs.open("Connect")
	
	def set_title(self, name = None):

		title = "CeraMatch"
		if name is None:
			self.setWindowTitle(title)
		else:
			self.setWindowTitle("%s - %s" % (name, title))
	
	def show_progress(self, text):
		
		self.progress = QtWidgets.QProgressDialog(text, None, 0, 0, self, flags = QtCore.Qt.FramelessWindowHint)
		self.progress.setWindowModality(QtCore.Qt.WindowModal)
		self.progress.show()
		QtWidgets.QApplication.processEvents()
	
	def hide_progress(self):
		
		self.progress.hide()
		self.progress.setParent(None)
	
	def update(self):
		
		self.set_title(os.path.split(str(self.model.identifier))[-1].strip("#"))
		self.calculate_button.setEnabled(self.model.has_samples() and (not self.model.has_distance()))
	
	def save(self):
		
		if self.model.data_source is None:
			self.dialogs.open("Connect")
			
		else:
			self.show_progress("Saving...")
			self.model.save()
			self.hide_progress()
	
	@QtCore.Slot()
	def on_load_data(self):
		
		self.show_progress("Loading...")
		if self.descriptor_group.lap_descriptors is not None:
			self.model.load_samples(self.descriptor_group.lap_descriptors)
		
		labels = dict([(str(sample_id), sample_id) for sample_id in self.model.sample_ids])
		self.graph_view.set_data(self.model.sample_data, self.model.sample_ids, [], labels = labels)
		# TODO load clustering if available
		
		self.descriptor_group.update()
		self.cluster_group.update_n_clusters()
		self.cluster_group.update_clusters_found(None)
		self.update()
		self.hide_progress()
	
	@QtCore.Slot()
	def on_cluster(self):
		
		self.show_progress("Clustering...")
		self.cluster_group.update_clusters_found(None)
		nodes, edges, clusters, labels = self.model.get_clusters(*self.cluster_group.get_limits())
		if nodes is None:
			return
		self.cluster_group.update_clusters_found(len(clusters))
		self.graph_view.set_data(self.model.sample_data, nodes, edges, clusters, labels)
		self.hide_progress()
	
	@QtCore.Slot()
	def on_calculate(self):
		
		self.show_progress("Calculating...")
		self.model.calc_distance()
		self.update()
		self.hide_progress()
	
	def on_view_action(self, *args):
		
		pass
	
	def on_broadcast(self, signals):
		
		if (Broadcasts.STORE_SAVED in signals) or (Broadcasts.STORE_SAVE_FAILED in signals):
			self.process_broadcasts()
		else:
			self.model.broadcast_timer.start(100)
	
	def on_broadcast_timer(self):

		self.process_broadcasts()
	
	def on_data_source_changed(self, *args):
		
		self.set_title(os.path.split(str(self.model.identifier))[-1].strip("#"))
		self.statusbar.message("")
	
	def on_data_changed(self, *args):
		
		self.statusbar.message("")
	
	def on_saved(self, *args):
		
		self.statusbar.message("Database saved.")
	
	def on_save_failed(self, *args):
		
		self.statusbar.message("Saving failed!")
		
