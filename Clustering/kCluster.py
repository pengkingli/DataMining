#-*-coding:utf-8-*-

import numpy as np
import math
import xlrd
import random
import matplotlib.pyplot as plt

def loadData(filename):
    data = xlrd.open_workbook(filename)
    table = data.sheets()[0]
    print "table: ", table
    nrows = table.nrows
    dataset = []
    for i in xrange(nrows):
        dataset.append(table.row_values(i))
    return dataset

def kMeans(data,k): 
    dataset = np.array(data)
    m,n = dataset.shape
    cluster = random.sample(dataset,k)
    #cluster = np.array(cluster)
    dic = {}
    dicbak = {}
    for time in xrange(500):
        dic = {}
        #print cluster
        for data in dataset:
            #print "666"
            minindex = minlength(data,cluster)
            if minindex not in dic.keys(): 
                dic[minindex] = []
            dic[minindex].append((np.mat(data)).tolist()[0]) #array先转化为mat,mat用tolist函数转化为list(注意一维数组的特殊情况)
        for index in range(k):
            #print dic[index]
            cluster[index] = np.array(dic[index]).mean(axis=0) #axis=0:按列求和
        #print dic
        if len(dic)!=0 and dic == dicbak:
            break
        dicbak = dic.copy()
    return cluster,dic
    
def minlength(inX,cluster):
    cluster = np.array(cluster)
    inX = np.array(inX)
    clm = cluster.shape[0]
    minindex = np.inf
    minlen = np.inf
    for i in xrange(clm):
        currlen = lengthcalc(inX,cluster[i])
        if currlen < minlen:
            minindex = i
            minlen = currlen
    return minindex
    
def lengthcalc(inX,inY): #inX,inY  要求同为行向量
    subdu = inX - inY
    subdu.shape = (1,subdu.shape[0])#一维数组转置必须指定大小
    return pow(np.dot(subdu,subdu.T),0.5)[0][0] #从类似于array([[ 0.09625487]])中取出值
    
def DBIcalc(cul,dic):
    dicset = np.array(dic)
    k = len(dic)
    kDBI = 0.0
    for i in range(k):
        maxsim = 0
        kj = range(k)
        del kj[i]
        for j in kj:
            sim = similar(dic[i],dic[j],cul[i],cul[j])
            #print sim
            if sim > maxsim:
                maxsim = sim
        #print 'maxsim',maxsim
        kDBI += maxsim
        #print 'kDBI',kDBI
    return 1.0/k*kDBI
    
def avgC(data): #输入为一个类的数据集
    dataset = np.array(data)
    k = dataset.shape[0]
    sumlength = 0.0
    for i in range(k):
        kj = range(k)
        del kj[i]
        for j in kj:
            sumlength += lengthcalc(dataset[i],dataset[j])
    return (2.0/(k*(k-1)))*sumlength

def similar(dataX,dataY,inX,inY): #输入为两个分类中的数据集,inX,inY为两个类的中心
    return float(avgC(dataX)+avgC(dataY))/lengthcalc(inX,inY)


def bestCluster(data):
    bestdbi = np.inf
    for k in range(2,5):
        cul,dic = kMeans(data,k)
        currdbi = DBIcalc(cul,dic)
        if currdbi < bestdbi:
            bestdbi = currdbi
    return k

    
def figplot(dic):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    col = ['r','b','g','k','y','m']
    for key in dic.keys():
        #ax.plot(np.array(dic[key])[:,0],np.array(dic[key])[:,1],linestyle='o--',color=col[key])
        ax.scatter(np.array(dic[key])[:,0],np.array(dic[key])[:,1],color=col[key])
    fig.show()
            