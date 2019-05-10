from PySide2 import (QtWidgets, QtCore, QtGui)

class ButtonsFrame(QtWidgets.QFrame):
	
	def __init__(self, view):
		
		self.view = view
		self.model = view.model
		
		QtWidgets.QFrame.__init__(self, view)
		
		self.setLayout(QtWidgets.QHBoxLayout())
		
#		button_css = "QPushButton {padding: 10px; font-size: 12pt;}"
		button_css = "QPushButton {padding: 10px;}"
		
		self.button_clustering = QtWidgets.QPushButton("Clustering")
		self.button_clustering.setStyleSheet(button_css)
		self.button_clustering.setCheckable(True)
		self.button_clustering.clicked.connect(self.view.on_clustering_mode)
		
		self.button_distance = QtWidgets.QPushButton("Distance")
		self.button_distance.setStyleSheet(button_css)
		self.button_distance.setCheckable(True)
		self.button_distance.clicked.connect(self.view.on_distance_mode)
		
		self.button_recalc = QtWidgets.QPushButton("Re-calculate")
		self.button_recalc.setStyleSheet(button_css)
		self.button_recalc.clicked.connect(self.view.on_recalc)
		
		self.button_ids = QtWidgets.QPushButton("Show IDs")
		self.button_ids.setStyleSheet(button_css)
		self.button_ids.clicked.connect(self.view.on_show_ids)
		
		self.slider_zoom = QtWidgets.QSlider(QtCore.Qt.Horizontal)
		self.slider_zoom.setMinimum(64)
		self.slider_zoom.setMaximum(256)
		self.slider_zoom.valueChanged.connect(self.view.on_zoom)
		
		self.layout().addWidget(self.button_clustering)
		self.layout().addWidget(self.button_distance)
		self.layout().addWidget(self.button_recalc)
		self.layout().addWidget(self.button_ids)
		self.layout().addStretch()
		self.layout().addWidget(QtWidgets.QLabel("Zoom:"))
		self.layout().addWidget(self.slider_zoom)
		
		self.update()
	
	def update(self):
		
		self.button_clustering.setChecked(not self.view.distance_mode_on())
		self.button_distance.setChecked(self.view.distance_mode_on())
		self.button_distance.setEnabled((len(self.view.image_lst.get_selected()) > 0) or self.view.distance_mode_on())
		
