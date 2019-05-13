
class Sample(object):
	
	def __init__(self, id, resource, label, value, row):
		
		self.id = id
		self.resource = resource
		self.label = label
		self.value = value
		self.row = row
		self.index = None
		self.cluster = None
		self.leaf = None

	def has_cluster(self):
		
		if self.cluster:
			return True
		return False

