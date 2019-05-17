
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
	
	def to_dict(self):
		
		return dict(
			id = self.id,
			value = self.value,
			cluster = self.cluster,
			leaf = self.leaf,
		)
	
	def from_dict(self, data):
		
		self.id = data["id"]
		self.value = data["value"]
		self.cluster = data["cluster"]
		self.leaf = data["leaf"]
