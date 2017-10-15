#!/usr/bin/python 
# -*- coding: utf-8 -*-

import sys
sys.path.append('D:\\Program Files\\Python2.7\\source\\DataMining\\Bayes')

import bayesi
import numpy as np
import pandas as pd
reload(bayesi)

filename = u'D:\\Program Files\\Python2.7\\source\\Watermelon\\Wm3.txt'

'''
DatRaw = bayesi.loadData(filename)  
Dat = bayesi.ChineseToNum(DatRaw)  
Data = np.array(Dat)
Data = np.reshape(Dat,(17,10))
'''
Data = bayesi.loadDatapd(filename)
Data = bayesi.toNum(Data)
Pc,Pcx,U = bayesi.bayes(Data)
data_pre = pd.DataFrame([[1,0,1,2,0,0,0.697,0.460]],columns= Data.columns[:-2])
bayesi.predict(data_pre,Pc,Pcx,U)
print U