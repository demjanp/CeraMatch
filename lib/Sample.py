
class Sample(object):
	
	def __init__(self, id, resource, label, value, row, obj_id):
		# label = text label or {clustering_level: color, ...}
		
		self.id = id
		self.resource = resource
		self.label = label
		self.value = value
		self.row = row
		self.obj_id = obj_id
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
			obj_id = self.obj_id,
		)
	
	def from_dict(self, data):
		
		self.id = data["id"]
		self.value = data["value"]
		self.cluster = data["cluster"]
		self.leaf = data["leaf"]
		self.obj_id = data["obj_id"]
