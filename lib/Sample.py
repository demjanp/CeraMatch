
class Sample(object):
	
	def __init__(self, id, resource, label, value, row):
		
		self.id = id
		self.resource = resource
		self.label = label
		self.value = value
		self.row = row
		self.index = None
		self.cluster = None
		self.cluster_color = None
		self.subcluster = None
