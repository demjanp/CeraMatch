from lib.ImageDelegate import ImageDelegate
from lib.IconThread import IconThread

from PySide2 import (QtWidgets, QtCore, QtGui)

class ListModel(QtCore.QAbstractListModel):
	
	icon_loaded = QtCore.Signal(QtCore.QModelIndex)
	
	def __init__(self, view, icon_size):
		
		self.view = view
		self.model = view.model
		self.icon_size = icon_size
		self.icons = [] # [QIcon or None, ...]; for each image
		self.paths = [] # [path or None, ...]; for each image
		self.empty_icon = None
		self.proxy_model = None
		self.threads = {} # {row: IconThread, ...}
		
		QtCore.QAbstractListModel.__init__(self)
		
		pixmap = QtGui.QPixmap(self.icon_size, self.icon_size)
		pixmap.fill()
		self.empty_icon = QtGui.QIcon(pixmap)
		
		self.font = QtGui.QFont()
		self.font.setPointSize(12)
		
		self.icons = [None] * len(self.model.samples)
		self.paths = [None] * len(self.model.samples)
	
	def stop_threads(self):
		
		for row in self.threads:
			self.threads[row].terminate()
			self.threads[row].wait()
		self.threads = {}
	
	def rowCount(self, parent):
		
		return len(self.model.samples)
	
	def flags(self, index):
		
		return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
		
	def data(self, index, role):
		
		if role == QtCore.Qt.DisplayRole:
			label = self.model.samples[index.row()].label
			if isinstance(label, dict):
				return None
			return label
		
		if role == QtCore.Qt.BackgroundRole:
			label = self.model.samples[index.row()].label
			if isinstance(label, dict):
				return label[max(list(label.keys()))]
			return QtGui.QColor(QtCore.Qt.white)
		
		if role == QtCore.Qt.DecorationRole:
			icon = self.icons[index.row()]
			if icon is None:
				return self.empty_icon
			return icon
		
		if role == QtCore.Qt.ToolTipRole:
			path = self.paths[index.row()]
			if path:
				return "<img src=\"%s\">" % (path)
			return ""
		
		if role == QtCore.Qt.UserRole:
			item = self.model.samples[index.row()]
			item.index = index
			return item
			
		return None
		
	def on_icon_thread(self, index, path):
		
		if not path is None:
			
			self.paths[index.row()] = path
			self.icons[index.row()] = QtGui.QIcon(path)
			self.icon_loaded.emit(index)
	
	def on_paint(self, index):
		
		if index is None:
			return
		
		row = index.row()
		if (self.icons[row] is None) and (not row in self.threads):
			self.threads[row] = IconThread(self, index, self.icon_size)
			self.threads[row].start()

class ImageList(QtWidgets.QListView):
	
	def __init__(self, view, icon_size = 256):
		
		self.view = view
		self.model = view.model
		self.icon_size = icon_size
		self.list_model = None
		self.index_lookup = {} # {sample_id: index, ...}
		
		QtWidgets.QListView.__init__(self, view)
		
		self.set_up()
		
	def set_up(self):
		
		self.list_model = ListModel(self.view, self.icon_size)
		
		self.setItemDelegate(ImageDelegate(self))
		
		self.setViewMode(QtWidgets.QListView.IconMode)
		self.setUniformItemSizes(True)
		self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
		self.setResizeMode(QtWidgets.QListView.Adjust)
		self.setWrapping(True)
		self.setFlow(QtWidgets.QListView.LeftToRight)
		
		self.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
		
		self.setModel(self.list_model)
		
		try: self.activated.disconnect()
		except: pass
		self.activated.connect(self.on_activated)
		
		self.list_model.icon_loaded.connect(self.on_icon_loaded)
		
		self.index_lookup = {}
		for row in range(self.list_model.rowCount(self)):
			index = self.list_model.index(row, 0)
			sample_id = index.data(QtCore.Qt.UserRole).id
			self.index_lookup[sample_id] = index
	
	def reload(self):
		
		self.list_model.stop_threads()
		self.set_up()
		self.view.update()
	
	def update_(self):
		
		pass
	
	def set_thumbnail_size(self, value):
		
		self.setIconSize(QtCore.QSize(value, value))
	
	def get_selected(self):
		# returns [Sample, ...]
		
		return [index.data(QtCore.Qt.UserRole) for index in self.selectionModel().selectedIndexes()]
	
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
		
		super(ImageList, self).selectionChanged(selected, deselected)
		self.model.on_selected()
		self.view.update()
	
	def on_icon_loaded(self, index):
		
		self.update(index)


