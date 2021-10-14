from lib.CeraMatchMain import (CeraMatchMain)
from multiprocessing import freeze_support
from os import system

if __name__ == '__main__':
	freeze_support()
	
	system("title CeraMatch")
	gui = CeraMatchMain()

# for pyinstaller:
import numpy as np
from scipy._lib.messagestream import *
import sklearn.utils._weight_vector
import skimage.filters.rank.core_cy_3d
