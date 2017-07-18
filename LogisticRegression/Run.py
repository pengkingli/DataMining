#!/usr/bin/python 
# -*- coding: utf-8 -*-

import LogReg
import numpy as np
import sys
sys.path.append('D:\\Program Files\\Python2.7\\source\\DataMining\\LogisticRegression')

reload(LogReg)

filename = r'D:\\Program Files\\Python2.7\\source\\DataMining\\LogisticRegression\\horseColicTraining.txt'
dataset = LogReg.LoadData(filename)
dataset = np.array(dataset) #299行，22列的数据
dat = dataset[:30,:]
w = LogReg.gradAscent(dat)
acc = LogReg.accuratecalc(dat,w)


filename = r'D:\\Program Files\\Python2.7\\source\\DataMining\\LogisticRegression\\horseColicTest.txt'
datatest = LogReg.LoadData(filename)
datatest = np.array(datatest) #299行，22列的数据
acc = LogReg.accuratecalc(datatest,w)