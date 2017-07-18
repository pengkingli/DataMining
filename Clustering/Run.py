#!/usr/bin/python 
# -*- coding: utf-8 -*-


import sys
sys.path.append('D:\\Program Files\\Python2.7\\source\\DataMining\\Clustering')
import kCluster
import numpy as np
import matplotlib.pyplot as plt

reload(kCluster)

filename = r'D:\\Program Files\\Python2.7\\source\\DataMining\\Clustering\\WM4.xlsx'
dataset = kCluster.loadData(filename)
dataset = np.array(dataset) 

cul,dic = kCluster.kMeans(dataset,3)
kCluster.figplot(dic)
