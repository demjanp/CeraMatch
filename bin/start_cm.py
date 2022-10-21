from ceramatch import CMMain

from multiprocessing import freeze_support
from os import system
import sys

if __name__ == '__main__':
	freeze_support()
	
	system("title CeraMatch")
	
	gui = CMMain()

