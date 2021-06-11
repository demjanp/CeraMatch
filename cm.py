from lib.CeraMatchMain import (CeraMatchMain)
from multiprocessing import freeze_support
from os import system

if __name__ == '__main__':
	freeze_support()
	
	system("title CeraMatch")
	gui = CeraMatchMain()

# for pyinstaller:
from sklearn.decomposition import *
#import sklearn.neighbors.typedefs
import numpy as np
from scipy._lib.messagestream import *
