from lib.toolbar._Tool import (Tool)

class Undo(Tool):
	
	def name(self):
		
		return "Undo"
	
	def icon(self):
		
		return "undo.svg"
	
	def help(self):
		
		return "Undo last action"
	
	def enabled(self):
		
		return self.view.graph_view.history.can_undo()
	
	def triggered(self, state):
		
		self.view.graph_view.undo()

