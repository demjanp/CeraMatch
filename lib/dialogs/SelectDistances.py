from lib.dialogs._Dialog import (Dialog)

from PySide2 import (QtWidgets, QtCore, QtGui)

class SelectDistances(Dialog):
	
	def title(self):
		
		return "Select Distances to Calculate"
	
	def set_up(self):
		
		self.setMinimumWidth(350)
		self.setModal(True)
		self.setLayout(QtWidgets.QVBoxLayout())
		
		self.check_diam = QtWidgets.QCheckBox("Diameter")
		self.check_diam.setChecked(True)
		self.check_ax = QtWidgets.QCheckBox("Axis")
		self.check_ax.setChecked(True)
		self.check_dice = QtWidgets.QCheckBox("Dice")
		self.check_dice.setChecked(True)
		self.check_dice_rim = QtWidgets.QCheckBox("Dice (rim only)")
		self.check_dice_rim.setChecked(True)
		
		self.layout().addWidget(self.check_diam)
		self.layout().addWidget(self.check_ax)
		self.layout().addWidget(self.check_dice)
		self.layout().addWidget(self.check_dice_rim)
		
	def process(self):
		
		diam, axis, dice, dice_rim = self.check_diam.isChecked(), self.check_ax.isChecked(), self.check_dice.isChecked(), self.check_dice_rim.isChecked()
		if True not in [diam, axis, dice, dice_rim]:
			return
		self.model.calc_distance(diam, axis, dice, dice_rim)
		self.view.update()

