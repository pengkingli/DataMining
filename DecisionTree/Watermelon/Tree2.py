
#-*- coding utf-8 -*-

import numpy as np

from math import log

def loadData(filename):
    DataSet = []
    fr = open(filename)
    for line in fr.readlines():
        curline = line.strip().split(',')
        DataSet.append(curline)
        return DataSet[1:]
        
    

'''
filename = source\Watermelon\Wm2.0

'''