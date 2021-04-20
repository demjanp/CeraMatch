
from copy import copy

class History(object):
	
	def __init__(self, clustering):
		
		self._clustering = clustering
		self._states = []
		self._idx = -1
		self._empty = [{}, set([]), set([]), {}, {}]
		
		# self._states = [[clusters, nodes, edges, labels, positions], ...]
		#	clusters = {#node_id: [@sample_id, ...], ...}
		#	nodes = [@sample_id, #node_id, ...]
		#	edges = [(@sample_id, #node_id), (#node_id1, #node_id2), ...]
		#	labels = {@sample_id: label, #node_id: label, ...}
		#	positions = {@sample_id: (x, y), #node_id: (x, y), ...}
	
	def clear(self):
		
		self._states = []
		self._idx = -1
	
	def can_undo(self):
		
		return (len(self._states) > 1) and ((self._idx == -1) or (self._idx > 0))
	
	def can_redo(self):
		
		return (len(self._states) > 0) and (self._idx > -1) and (self._idx < len(self._states) - 1)
	
	def undo(self):
		# returns clusters, nodes, edges, labels, positions
		
		if not self.can_undo():
			return self._empty
		if self._idx == -1:
			self._idx = len(self._states) - 2
		elif self._idx > 0:
			self._idx -= 1
		else:
			return self._empty
		return self._states[self._idx]
	
	def redo(self):
		# returns clusters, nodes, edges, labels, positions
		
		if not self.can_redo():
			return self._empty
		self._idx += 1
		return self._states[self._idx]
	
	def save(self):
		
		data = self._clustering.get_data()
		
		if self._states and (data == self._states[self._idx]):
			return
		
		if self._idx > -1:
			self._states = self._states[:self._idx]
			self._idx = -1
		
		if self._idx >= len(self._states) - 1:
			self._idx = -1
		
		self._states.append(data)
	
	