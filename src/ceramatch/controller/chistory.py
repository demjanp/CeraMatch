from PySide2 import (QtCore)

class CHistory(QtCore.QObject):
	
	def __init__(self, cmain):
		
		QtCore.QObject.__init__(self)
		
		self.cmain = cmain
		self._history = []
		self._index = 0
		self._paused = False
	
	def can_undo(self):
		
		return self._index > 0
	
	def can_redo(self):
		
		return self._index < len(self._history) - 1
	
	def clear(self):
		
		self._history.clear()
		self._index = 0
	
	def pause(self):
		
		self._paused = True
	
	def resume(self):
		
		self._paused = False
	
	def save(self):
		
		if self._paused:
			return
		
		data = self.cmain.cgraph.get_cluster_data()
		self._history = self._history[:self._index + 1]
		if self._history and (data == self._history[-1]):
			return
		
		self._history.append(data)
		self._index = len(self._history) - 1
		self.cmain.cactions.update()
	
	def undo(self):
		
		if self._index < 1:
			return
		self.pause()
		self._index -= 1
		self.cmain.cgraph.set_clusters(*self._history[self._index])
		self.resume()
		self.cmain.cactions.update()

	
	def redo(self):
		
		if self._index > len(self._history) - 1:
			return
		self.pause()
		self._index += 1
		self.cmain.cgraph.set_clusters(*self._history[self._index])
		self.resume()
		self.cmain.cactions.update()

