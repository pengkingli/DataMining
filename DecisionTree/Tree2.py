
'''#-*- coding utf-8 -*-'''
#决策树算法实现，使用的数据集为西瓜数据集2.0，包含了后剪枝算法
# coding=gbk
#以utf-8-BOM进行编码
import numpy as np
import operator
from math import log

#导入数据
def loadData(filename):
    DataSet = []
    #考虑不在数据集中“是/否”之后加，如何正常导入数据集，避免\n和之后的编号在一起
    DataSet = open(filename).read().split(',') #读入为一个字符串并以','进行切分; 但是没能将行末和下一行的开头区分开
    return DataSet

#由于得到的为变卡的字符串，通过该函数将字符串转化为数字，便于处理    
def ChineseToNum(DatSet):
    NumDat = []
    for dat in DatSet:
        if unicode(dat,"gbk") == u'浅白':
            NumDat.append(0)
        elif unicode(dat,"gbk") == u'青绿':
            NumDat.append(1)
        elif unicode(dat,"gbk") == u'乌黑':
            NumDat.append(2)
        elif unicode(dat,"gbk") == u'蜷缩':
            NumDat.append(0)
        elif unicode(dat,"gbk") == u'稍蜷':
            NumDat.append(1)
        elif unicode(dat,"gbk") == u'硬挺':
            NumDat.append(2)
        elif unicode(dat,"gbk") == u'沉闷':
            NumDat.append(0)
        elif unicode(dat,"gbk") == u'浊响':
            NumDat.append(1)
        elif unicode(dat,"gbk") == u'清脆':
            NumDat.append(2)
        elif unicode(dat,"gbk") == u'模糊':
            NumDat.append(0)
        elif unicode(dat,"gbk") == u'稍糊':
            NumDat.append(1)
        elif unicode(dat,"gbk") == u'清晰':
            NumDat.append(2)
        elif unicode(dat,"gbk") == u'凹陷':
            NumDat.append(0)
        elif unicode(dat,"gbk") == u'稍凹':
            NumDat.append(1)
        elif unicode(dat,"gbk") == u'平坦':
            NumDat.append(2)
        elif unicode(dat,"gbk") == u'硬滑':
            NumDat.append(0)
        elif unicode(dat,"gbk") == u'软粘':
            NumDat.append(1)
        elif unicode(dat,"gbk") == u'是':
            NumDat.append(1)
        elif unicode(dat,"gbk") == u'否':
            NumDat.append(0)
        else:
            #print unicode(dat,"gbk")
            pass
    #print len(NumDat)
    return NumDat

#计算香农信息熵
def InfoEntCalc(Label):
    LabelNum = len(Label)
    labelCount = {}
    ShannonEnt = 0.0
    for currentlabel in Label:
        if currentlabel not in labelCount.keys():
            labelCount[currentlabel] = 0
        labelCount[currentlabel] += 1
    for key in labelCount.keys():
        EntPk = labelCount[key]/(LabelNum*1.0)
        ShannonEnt -= EntPk*log(EntPk,2)
    return ShannonEnt

#采用ID3算法，即用信息增益作为选择划分属性的标准
def InfoGain(DatSet,Label,k):   #k为第k个特征
    m,n = np.shape(DatSet)
    ShannonEntfore = InfoEntCalc(Label)
    DatSet = DatSet[:,k]
    labelCount = {}
    subShannonEnt = 0.0
    for currentlabel in DatSet:
        if currentlabel not in labelCount.keys():
            labelCount[currentlabel] = 0
        labelCount[currentlabel] += 1
    for key in labelCount.keys():
        subDataSet = Label[DatSet==key]
        subShannonEnt += labelCount[key]/(m*1.0)*InfoEntCalc(subDataSet)
        #print key,labelCount[key],labelCount[key]/(m*1.0)*InfoEntCalc(subDataSet)
    return ShannonEntfore - subShannonEnt

#得到最大增益的属性
def MaxGain(DatSet,Label):
    m,n = np.shape(DatSet)  #和其他函数综合起来看，有一些重复计算
    Gain = 0.0
    maxGain = -1
    bestFeature = -1
    for featureNum in range(n):
        Gain = InfoGain(DatSet,Label,featureNum)
        if Gain > maxGain:
            bestFeature = featureNum
            maxGain = Gain
    return bestFeature

def majorCnt(DatSet):   #当前数据集返回类别数目最多的特征，借鉴了机器学习实战
    Label = DatSet[:]
    LabelCnt = {}
    for value in Label:
        if value not in LabelCnt.keys():
            LabelCnt[value] = 0
        LabelCnt[value] += 1
        sortedLabelCnt = sorted(LabelCnt.iteritems(),key = operator.itemgetter(1),reverse=True)
        return sortedLabelCnt[0][0]
        

#基本的决策树构建
def TreeGenerate(Dat,DatOri,Table):  #输入位np array格式
    DatSet = Dat[:,:-1]  #取出所有的数据集
    Label = Dat[:,-1]   #取出样本对应得类别集
    m,n = np.shape(DatSet)
    #当所有数据集的分类相同时：
    #if( (m == sum(Label)/Label[0]) or sum(Label) ==0) #
    if list(Label).count(Label[0]) == m:
        return Label[0]
    #属性集已经遍历完成，但是数据中仍然有不同的分类，即最后一个属性中既有好瓜也有坏瓜
    if n == 1:  #n=1表示只剩下了1个类别，
        return majorCnt(Label)
    #if len(DatSet) == 0:
    bestFeature = MaxGain(DatSet,Label) #bestFeature：最大增益特征对应特征的编号
    #feature = Table[bestFeature] 
    #print bestFeature
    bestFeatureTable = Table[bestFeature]#根据编号选出特征
    #print bestFeatureTable
    #print Table
    Tree = {bestFeatureTable:{}}
    del(Table[bestFeature]) #用过的特征要删除
    #print Table
    #print bestFeatureTable,set(DatOri[:,bestFeature])
    for value in set(DatOri[:,bestFeature]): #对属性的每个结果
        #print (bestFeatureTable,value)
        subDatSetR = Dat[Dat[:,bestFeature] == value] #选出属性bestFeature，值为value的行
        subDatSet = np.concatenate((subDatSetR[:,:bestFeature],subDatSetR[:,bestFeature+1:]),axis=1) #数据集将bestFeature特征去掉，并选出特征值为value的数据集
        subDatOri = np.concatenate((DatOri[:,:bestFeature],DatOri[:,bestFeature+1:]),axis=1) #subDatOri：数据集之将bestFeature属性去掉。不区分特征的取值
        subTabel = Table[:]
        subm,subn = np.shape(subDatSet)
        #print subm
        #print "Label:", Label
        if(subm == 0):  #当子集的数据集为空时，说明没有这样的特征样本，根据其父集中样本最多的类作为其类别
            Tree[bestFeatureTable][value] = majorCnt(Label)#return majorCnt(Label)
        else:
            Tree[bestFeatureTable][value] = TreeGenerate(subDatSet,subDatOri,subTabel)  #Tree[bestFeature][value]两层深度的树
    return Tree
    
    
#测试函数
def Classify(inputTree,featureTable,testDatSet): 
    testData = np.array(testDatSet)
    key = inputTree.keys()[0]  #取出决策树的划分属性
    selectfeature = featureTable.index(key) #根据属性得到其对应的索引号
    featurevalue = testDatSet[selectfeature]
    subTree = inputTree[key]
    for key in subTree.keys():
        if featurevalue == key: #
            if type(subTree[key]).__name__ == 'dict':
                classLabel = Classify(subTree[key],featureTable,testDatSet)
            else: classLabel = subTree[key]
    return classLabel

    
#决策树剪枝，采用后剪枝的方法，但是该函数支队最后一层进行剪枝，中间层不剪枝
#主要思路是：遍历已经建成的树的所有最后一层树，对其进行后剪枝处理，处理后得到一颗新的树返回，而不是在原树上进行操作
def PostPurn(Tree,featureTable,trainData,testData):
    firstkey = Tree.keys()[0]
    subTree = Tree[firstkey]
    Tree3 = {firstkey:{}}
    #firstselectfeature = featureTable.index(firstkey) #根据特征得到其对应的索引号
    for key in subTree.keys():
        #print "key= ",(firstkey,key)
        selectfeature = featureTable.index(firstkey) #根据特征得到其对应的索引号
        subtestData = testData[testData[:,selectfeature] == key]
        subtrainData = trainData[trainData[:,selectfeature] == key]
        subtrainLabel = subtrainData[:,-1]
        sub2Tree = subTree[key]
        #print 'subTree[key]: ',subTree[key]
        if type(subTree[key]).__name__ == 'int':
            Tree3[firstkey][key] = subTree[key]
        else:
            if isendTree(subTree[key]):  #如果是最后一层树，即所有的节点均为值而不是字典
                Tree2 = subTree.copy()
                #print 'subtrainLabel',subtrainLabel
                Tree2[key] = majorCnt(subtrainLabel) #结果为1个值
                #print Tree2[key]
                Accurateafter = AccurateCalcnotTree(Tree2[key],featureTable,subtestData)
                Accuratebefore = AccurateCalc(Tree,featureTable,subtestData)
                if Accurateafter > Accuratebefore: #判断剪枝前后的验证集精度
                    Tree3[firstkey][key] = Tree2[key]
                else: 
                    Tree3[firstkey][key] = subTree[key]
            else: Tree3[firstkey][key] = PostPurn(sub2Tree,featureTable,subtrainData,subtestData)
    return Tree3

def AccurateCalc(Tree,featureTable,testData): #testData为np.array格式，对数进行数据集精度计算
    testDat = testData[:,:-1]
    testLable = testData[:,-1]
    m,n = np.shape(testDat)
    #rightnum = 0.0
    rightcounter = 0
    for num in range(m):
        #print Classify(Tree,featureTable,testDat[num])
        if(Classify(Tree,featureTable,testDat[num]) == testLable[num]):
            rightcounter += 1
            #print rightcounter
    return rightcounter/(m*1.0)

def AccurateCalcnotTree(Tree,featureTable,testData): #所有节点均为根节点的树，数据集精度计算
    testDat = testData[:,:-1]
    testLable = testData[:,-1]
    m,n = np.shape(testDat)
    rightcounter = 0
    for num in range(m):
        if(Tree == testLable[num]):
            rightcounter += 1
            #print rightcounter
    return rightcounter/(m*1.0)
    
def isTree(obj): #判断是否为一棵树
    return (type(obj).__name__ == 'dict')
    
def isendTree(Tree): #判断是否是所有节点均为根节点的树,Tree为一个字典
    #if type(Tree).__name__ == 'dict':
    #    return True
    subTree = Tree.values()[0]
    if type(subTree).__name__ == 'dict':
        for value in subTree.values():
            if isTree(value):
                return False
        return True
    else: return True
    
    
'''
filename = 'source/Watermelon/Wm2.txt'
Dat = Tree2.loadData(filename)
Datn = Tree2.ChineseToNum(Dat)
Dat =np.array(Datn)
Dat = np.reshape(Dat,(17,7))
DatSet = Dat[:,:-1]
DatTrain = Dat[[0,1,2,3,5,6,9,13,14,15,16]]#选出第...行的数据
DatTest = Dat[[3,4,7,8,10,11,12]]
Label = Dat[:,-1] #结果为一行向量
Tree2.InfoEntCalc(Label)
Tree2.InfoGain(DatSet,Label,0)
Tree2.MaxGain(DatSet,Label)

Table = [u'色泽',u'根蒂',u'敲声',u'纹理',u'脐部',u'触感'] #特征的集合
Table = ['A','B','C','D','E','F'] #特征的集合
reload(Tree2)
Tree2.TreeGenerate(Dat,Table)
testData = np.array([[1,2,2,1,2,1,0],[0,0,1,2,0,1,0],[2,1,2,0,1,1,1],[1,1,1,1,0,1,0],[2,0,1,2,1,1,0]])
Tree2.Classify(tree,Table,testData)
Tree2.AccurateCalc(tree,Table,testData)
Tree2.PostPurn(Tree,Table,DatTest)
Tree2.majorCnt(testData)
Tree2.PostPurn(Tree,Table,DatTrain)
''' 
    
'''

'''
            
    
    
    
    
    
'''
filename = 'source\Watermelon\Wm2.0'
D安图妮= 
print unicode(dat,"gbk")
print u'稍凹'.encode("gbk").decode("gbk")
print u'\u7eb9\u7406' #可以直接输出中文
'''