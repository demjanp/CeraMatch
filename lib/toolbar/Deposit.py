from lib.toolbar._Tool import (Tool)

class Deposit(Tool):
	
	def name(self):
		
		return "Open Deposit"
	
	def icon(self):
		
		return "dep_cube.svg"
	
	def help(self):
		
		return "Open Database in Deposit"
	
	def enabled(self):
		
		return True
	
	def triggered(self, state):
		
		self.model.launch_deposit()

