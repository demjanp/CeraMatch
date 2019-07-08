from lib.ImageDelegate import ImageDelegate
from lib.IconThread import IconThread

from PySide2 import (QtWidgets, QtCore, QtGui)

class TableModel(QtCore.QAbstractTableModel):
	
	icon_loaded = QtCore.Signal(QtCore.QModelIndex)
	
	def __init__(self, view, icon_size):
		
		self.view = view
		self.model = view.model
		self.icon_size = icon_size
		self.icons = [] # [QIcon or None, ...]; for each image
		self.paths = [] # [path or None, ...]; for each image
		self.empty_icon = None
		self.proxy_model = None
		self.threads = {} # {column: IconThread, ...}
		
		QtCore.QAbstractTableModel.__init__(self)
		
		pixmap = QtGui.QPixmap(self.icon_size, self.icon_size)
		pixmap.fill()
		self.empty_icon = QtGui.QIcon(pixmap)
		
		self.font = QtGui.QFont()
		self.font.setPointSize(12)
		
		self.icons = [None] * len(self.model.samples)
		self.paths = [None] * len(self.model.samples)
	
	def stop_threads(self):
		
		for column in self.threads:
			self.threads[column].terminate()
			self.threads[column].wait()
		self.threads = {}
	
	def rowCount(self, parent):
		
		return self.model.max_clustering_level()
	
	def columnCount(self, parent):
		
		return len(self.model.samples)
	
	def flags(self, index):
		
		return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
		
	def data(self, index, role):
		
		if role == QtCore.Qt.DisplayRole:
			return ""
			
		if role == QtCore.Qt.BackgroundRole:
			label = self.model.samples[index.column()].label
			level = index.row() + 1
			if isinstance(label, dict) and (level in label): # label = {clustering_level: color, ...}
				return label[level]
			return QtGui.QColor(QtCore.Qt.white)
			
		if role == QtCore.Qt.DecorationRole:
			label = self.model.samples[index.column()].label
			level = index.row() + 1
			if isinstance(label, dict) and (level in label):
				return self.icons[index.column()]
			return self.empty_icon
			
		if role == QtCore.Qt.ToolTipRole:
			path = self.paths[index.column()]
			if path:
				return "<img src=\"%s\">" % (path)
			return ""
			
		if role == QtCore.Qt.UserRole:
			item = self.model.samples[index.column()]
			item.index = index
			return item
			
		return None
		
	def on_icon_thread(self, index, path):
		
		if not path is None:
			
			self.paths[index.column()] = path
			self.icons[index.column()] = QtGui.QIcon(path)
			self.icon_loaded.emit(index)
	
	def on_paint(self, index):
		
		if index is None:
			return
		
		column = index.column()
		if (self.icons[column] is None) and (not column in self.threads):
			self.threads[column] = IconThread(self, index, self.icon_size)
			self.threads[column].start()

class ImageTable(QtWidgets.QTableView):
	
	def __init__(self, view, icon_size = 256):
		
		self.view = view
		self.model = view.model
		self.icon_size = icon_size
		self.zoomed_icon_size = icon_size
		self.list_model = None
		self.index_lookup = {} # {sample_id: index, ...}
		
		QtWidgets.QTableView.__init__(self, view)
		
		self.set_up()
		
	def set_up(self):
		
		self.list_model = TableModel(self.view, self.icon_size)
		
		self.setItemDelegate(ImageDelegate(self))
		self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
		self.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
		self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
		self.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
		self.horizontalHeader().hide()
		self.verticalHeader().hide()
		self.set_thumbnail_size(self.zoomed_icon_size)
		
		self.setModel(self.list_model)
		
		try: self.activated.disconnect()
		except: pass
		self.activated.connect(self.on_activated)
		
		self.list_model.icon_loaded.connect(self.on_icon_loaded)
		
		self.index_lookup = {}
		for column in range(self.list_model.columnCount(self)):
			index = self.list_model.index(0, column)
			sample_id = index.data(QtCore.Qt.UserRole).id
			self.index_lookup[sample_id] = index
	
	def reload(self):
		
		self.list_model.stop_threads()
		self.set_up()
		self.view.update()
	
	def update_(self):
		
		pass
	
	def set_thumbnail_size(self, value):
		
		self.zoomed_icon_size = value
		self.setIconSize(QtCore.QSize(value, value))
		self.horizontalHeader().setDefaultSectionSize(value*1.1)
		self.verticalHeader().setDefaultSectionSize(value*1.1)
	
	def get_selected(self):
		# returns [Sample, ...]
		
		return [index.data(QtCore.Qt.UserRole) for index in self.selectionModel().selectedIndexes()]
	
	def get_selected_level(self):
		# returns [level, ...]
		
		return [index.row() + 1 for index in self.selectionModel().selectedIndexes()]
	
	def set_selected(self, sample_ids):		
		
		self.blockSignals(True)
		for sample_id in sample_ids:
			self.selectionModel().select(self.index_lookup[sample_id], QtCore.QItemSelectionModel.SelectionFlag.Select)
		self.blockSignals(False)
	
	def on_activated(self, index):
		
		sample = index.data(QtCore.Qt.UserRole)
		
		text = "Sample ID\t%s\nValue\t%s" % (sample.id, sample.value)
		
		data = QtCore.QMimeData()
		data.setData("text/plain", bytes(text, "utf-8"))
		cb = QtWidgets.QApplication.clipboard()
		cb.setMimeData(data)
		self.view.statusbar.message("Data copied to clipboard")
	
	def selectionChanged(self, selected, deselected):
		
		super(ImageTable, self).selectionChanged(selected, deselected)
		self.model.on_selected()
		self.view.update()
	
	def on_icon_loaded(self, index):
		
		self.update(index)


