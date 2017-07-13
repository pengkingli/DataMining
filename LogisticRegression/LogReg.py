#-*- coding: UTF-8 -*- 
'''线性回归模型,使用的数据是《机器学习实战》中疝气病症预测马的死亡率预处理后的数据集'''
import numpy as np
import math
from collections import Counter

def LoadData(filename):
    dataset = []
    with open(filename) as file_object: #将数据读入内存，with会在合适的时候关闭文件
        file_lines = file_object.readlines()
    for line in file_lines:
        #print line
        dataset.append(line.strip().split())
    return dataset
    
    
def gradAscent(dat):
    dataset = dat[:,:-1]
    label = dat[:,-1]  #转置变为列向量
    dataset = dataset.astype(np.float64)
    label = label.astype(np.float64)
    #ml = label.shape
    label.shape = (1,label.shape[0])#一维数组转置必须指定大小
    label = label.T
    m,n = np.shape(dataset)
    oneline = np.ones((m,1))
    dataset = np.concatenate((dataset,oneline),axis=1)
    w = np.ones((n+1,1))
    #print dataset
    alpha = 0.01
    for num in xrange(100):
        #print w
        h = Sigmod(np.dot(dataset,w))
        #print np.dot(dataset,w)
        err = label - h
        #print 'label: ',label
        #print 'h: ',h
        #print 'err:',err
        w = w + alpha * np.dot(dataset.T,err) 
        #print 'w',w
    return  w
    
def resultcalc(inX,w):  #输入单个列向量，计算输出结果
    return Sigmod(np.array(inX).T * w[:,-1] +w[-1])
    
def Sigmod(z):
    return 1.0/(1 + math.e**(-z))
    
def fliter(inLab):
    label = inLab[:]
    for i in range(len(label)):
        if label[i] > 0.5:
            label[i] = 1
        else: label[i] = 0
    return label
    
    
def accuratecalc(dat,w):
    dataset = dat[:,:-1]
    label = dat[:,-1]  #转置变为列向量
    label.shape = (1,label.shape[0])#一维数组转置必须指定大小
    label = label.T  #转置变为列向量
    m,n = np.shape(dataset)
    oneline = np.ones((m,1))
    dataset = np.concatenate((dataset,oneline),axis=1)
    dataset = dataset.astype(np.float64)
    label = label.astype(np.float64)
    h = np.dot(dataset,w)
    #print Sigmod(h)
    labelcalc = fliter(Sigmod(h))
    err = label - labelcalc
    #count = Counter(list(err)),  Counter不能对numpy计数
    #print label,labelcalc
    rightcounter = np.count_nonzero(err)
    return 1 - rightcounter/(m * 1.0)