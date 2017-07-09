#!/usr/bin/python 
# -*- coding: utf-8 -*-
import Tree2
import numpy as np
reload(Tree2)

filename = 'source/Watermelon/Wm2.txt' #
Dat = Tree2.loadData(filename)
Datn = Tree2.ChineseToNum(Dat)
Dat =np.array(Datn)
Dat = np.reshape(Dat,(17,7))
DatSet = Dat[:,:-1]
Label = Dat[:,-1] #结果为一行向量
Tree2.InfoEntCalc(Label)
Tree2.InfoGain(DatSet,Label,0)
Tree2.MaxGain(DatSet,Label)
DatTrain = Dat[[0,3,1,2,5,6,9,13,14,15,16]]#选出第...行的数据
DatTest = Dat[[3,4,7,8,10,11,12]]
Table = [u'色泽',u'根蒂',u'敲声',u'纹理',u'脐部',u'触感'] #特征的集合
Table = ['A','B','C','D','E','F'] #特征的集合
reload(Tree2)
Tree2.TreeGenerate(Dat,Table)
Tree2.PostPurn(Tree,Table,DatTrain,DatTest)
Tree = {'E': {0: {'A': {0: 0, 1: 1, 2: 1}}, 1: {'B': {0: 0, 1: {'A': {1: 1, 2: {'D': {1: 1, 2: 0}}}}}}, 2: 0}}

Tree = {'E': {0: {'A': {0: 0, 1: 1, 2: 1}}, 1: {'B': {0: 0, 2:1,1: {'A': {1: 1,0:1, 2: {'D': {0:1,1: 1, 2: 0}}}}}}, 2: 0}}

import Tree_WM3
reload(Tree_WM3)
import numpy as np
filename = 'source/Watermelon/Wm3.txt'
Dat = Tree_WM3.loadData(filename)
Datn = Tree_WM3.ChineseToNum(Dat)
Dat =np.array(Datn)
Dat = np.reshape(Dat,(17,10))
DatSet = Dat[:,1:] #去除第一列标号
#好像数组内不能有两种格式，所以一下的转换不能达到前一部分是int型，后一部分是float型的目的
DatSet = np.concatenate((DatSet[:,:-2].astype(np.int),DatSet[:,-2:]),axis=1)
Label = Dat[:,-1] #结果为一行向量
Table = ['A','B','C','D','E','F','1','2'] #特征的集合
Tree_WM3.InfoEntCalc(Label)
Tree_WM3.InfoGain(DatSet,Label,0)
Tree_WM3.MaxGain(DatSet,Label)
Tree_WM3.InfoGainContous(DatSet,Label,6)
Tree_WM3.TreeGenerate(DatSet,DatSet,Table)
'''
class First(object):
    def __init__(self,nm,sc):
        self.name = nm
        self.score = sc
    def __pr(self):
        print self.name,self.score

bt = First('666',233)
#bt.se('666',233)
bt._First__pr()
        
class Student(object):
    def __init__(self, name, score):
        self.name = name
        self.score = score
bart = Student('Bart Simpson', 59)
print bart.name

c=666

def story(a,b=333):
    global d
    print a+b+globals()['c']
    d=9.9
#    print c,d 
    
    
story(1,2)
print d


import matplotlib.pyplot as plt
import numpy as np

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(2,2,1)
rn = np.random.random()
plt.plot(rn,'g--')


import svmMLiA
reload(svmMLiA)
dataArr,labelArr = svmMLiA.loadDataSet('source/testSetsvm.txt')
b,alphas = svmMLiA.smoP(dataArr,labelArr,0.6,0.001,40)



import svmMLiA
dataArr,labelArr = svmMLiA.loadDataSet('source/testSetsvm.txt')
b,alphas = svmMLiA.smoSimple(dataArr,labelArr,0.6,0.001,40)
print b



import numpy as np
import panda as pd
from pandas import Series, DataFrame

data = DataFrame(np.arange(16).reshape(4,4),index=list('abcd'),columns=list('1234'))
data

from pandas import*
df1 = DataFrame({'key':['b','b','a','c','a','b'],'data1':range(6)})
df2 = DataFrame({'key':['a','b','a','b','d'],'data2':range(5)})




import logRegres
dataArr,labelMat = logRegres.loadDataSet()
lg = logRegres.gradAscent(dataArr,labelMat)
lg

sd = {'a':1,'b':2,'c':3}
for key in sd:
    print key,



arr=np.random.randn(5,5)

print arr

print arr[::2].sort(1)



nwalks = 10
nstep  = 1000
draws  = np.random.randint(0,2,size=(nwalks, nstep))
steps  = np.where(draws > 0,1,-1)

walks  = steps.cumsum(1)

print walks



arr = np.random.randn(5,3)

#print arr
print arr.sort(1)


xo=np.array([1,2,3,4,5])
yo=np.array([2,4,6,8,10])
co=np.array(True,False,True,True,False)

result [(x if c else y)
         for x,y,c in zip(xo,yo,co]



names = np.array(['A','B','C','A','D'])
data =[[1,2],
        [2,3],
        [3,4],
        [4,5],
        [5,6]]

        
da1 = np.random.randn(7)*5
#da1 = data(names=='A')

print np.ceil(da1,5)
'''