from lib.toolbar._Tool import (Tool)

class Redo(Tool):
	
	def name(self):
		
		return "Redo"
	
	def icon(self):
		
		return "redo.svg"
	
	def help(self):
		
		return "Redo last undone action"
	
	def enabled(self):
		
		return self.view.graph_view.history.can_redo()
	
	def triggered(self, state):
		
		self.view.graph_view.redo()

