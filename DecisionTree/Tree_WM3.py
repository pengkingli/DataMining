
'''#-*- coding utf-8 -*-'''
# coding=gbk
#以utf-8-BOM进行编码
import numpy as np
import operator
from math import log

def loadData(filename):
    DataSet = []
    #考虑不在数据集中“是/否”之后加，如何正常导入数据集，避免\n和之后的编号在一起
    DataSet = open(filename).read().split(',') #读入为一个字符串并以，进行切分; 但是没能将是和之后的数字区分开
    '''
    for line in fr.readlines():
        curline = line.strip().split('\t')
        DataSet.append(line)
        return DataSet
    '''
    
    return DataSet
        
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
            try:
                float(dat)
                #if type(float(dat)).__name__ == 'float':
                NumDat.append(float(dat))
            except: pass
    print len(NumDat)
    return NumDat
    
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

    
def InfoGain(DatSet,Label,k):   #k为第k个特征  ID3用信息增益作为标准
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

def InfoGainContous(DatSet,Label,k):
    DatSetk = DatSet[:,k]
    nk = len(DatSetk)
    uniqueDatSetk = list(set(DatSetk))#set不能用索引获取值
    uniquesortDatSetk = np.sort(uniqueDatSetk)
    n = len(uniquesortDatSetk) #对于set用len方法,set无序
    selectPoint = []
    for index in range(n-1):
        #print index
        selectPoint.append((uniquesortDatSetk[index] + uniquesortDatSetk[index + 1])/2.0)
    #print 'selectPoint: ',selectPoint
        maxinfoEnt = 0.0
        bestPoint = -1
        bestLabel = []
        maxGain = 0
        #print 'Label: ',Label
    for index in range(n-1):
        Label0 = []  #用于存放小于划分点的值
        Label1 = []  #用于存放大于划分点的值
        labelCount = 0
        infoEnt = 0.0
        for datindex in range(nk):
            if DatSetk[datindex] < selectPoint[index]:
                labelCount += 1
                Label0.append(Label[datindex])
            else: Label1.append(Label[datindex])
        sumEnt = len(Label0)/(len(Label)*1.0)*InfoEntCalc(Label0) + len(Label1)/(len(Label)*1.0)*InfoEntCalc(Label1)
        infoEnt = InfoEntCalc(Label) - sumEnt
        #print Label0,len(Label0)/(len(Label)*1.0)*InfoEntCalc(Label0)
        #print Label1,len(Label1)/(len(Label*1.0))*InfoEntCalc(Label1)
        #print infoEnt,maxinfoEnt
        #print InfoEntCalc(Label),sumEnt
        if infoEnt > maxinfoEnt:
            maxinfoEnt = infoEnt
            bestPoint = selectPoint[index] #得到最佳划分点
            bestLabel = Label0
    return maxinfoEnt,bestPoint
    

def MaxGain(DatSet,Label,Table):
    m,n = np.shape(DatSet)  #多了一些重复计算
    Gain = 0.0
    maxGain = -1
    bestFeature = -1
    bestPoint = -1
    for tab in Table:
        featureNum = list(Table).index(tab)
        #print "featureNum: ",featureNum
        try: 
            float(tab)
        except:
            #print 'DatSet,Num',(DatSet,featureNum)
            Gain = InfoGain(DatSet,Label,featureNum)
            Point = -1
        else: 
            #print featureNum,Label,DatSet
            #print  "featureNum: ",featureNum
            Gain,Point = InfoGainContous(DatSet,Label,featureNum)
        #if featureNum == 6 or featureNum == 7:
        #    Gain,bestPoint = InfoGainContous(DatSet,Label,featureNum)
        #else: Gain = InfoGain(DatSet,Label,featureNum)
        if Gain > maxGain:
            bestFeature = featureNum
            maxGain = Gain
            bestPoint = Point
        #print featureNum,Gain
    return bestFeature,bestPoint

def majorCnt(DatSet):   #当前数据集返回类别数目最多的特征
    Label = DatSet[:]
    LabelCnt = {}
    for value in Label:
        if value not in LabelCnt.keys():
            LabelCnt[value] = 0
        LabelCnt[value] += 1
        sortedLabelCnt = sorted(LabelCnt.iteritems(),key = operator.itemgetter(1),reverse=True)
        return sortedLabelCnt[0][0]
        

#完成基本的决策树构建
def TreeGenerate(Dat,DatOri,Table):  #输入位np array格式
    DatSet = Dat[:,:-1]  #取出所有的数据集
    Label = Dat[:,-1]   #取出样本对应得类别集
    Tables = Table[:]
    m,n = np.shape(DatSet)
    #当所有数据集的分类相同时：
    if list(Label).count(Label[0]) == m:
        return Label[0]
    #属性集已经遍历完成，但是数据中仍然有多个分类类别时
    if n == 1:  #n=1表示只剩下了类别
        return majorCnt(Label)
    #if len(DatSet) == 0:
    #print DatSet,Label,Table
    bestFeature,bestPoint = MaxGain(DatSet,Label,Table) #bestFeature对应特征的编号
    #feature = Table[bestFeature] #根据编号选出特征字符串
    #print bestFeature,bestPoint
    bestFeatureTable = Table[bestFeature]
    #print bestFeatureTable
    #print bestFeatureTable
    #print Table
    del(Table[bestFeature])
    #print Table
    Tree = {bestFeatureTable:{}}
    try:
        int(bestFeatureTable)#根据选出的属性是否可以转化为int型确定是否为密度和含糖量
    except:  
    #print Table
    #print bestFeatureTable,set(DatOri[:,bestFeature])
        for value in set(DatOri[:,bestFeature]):
            #print (bestFeatureTable,value)
            subDatSetR = Dat[Dat[:,bestFeature] == value] #选出属性bestFeature，值为value的行
            subDatSet = np.concatenate((subDatSetR[:,:bestFeature],subDatSetR[:,bestFeature+1:]),axis=1) #数据集将bestFeature属性去掉
            subDatOri = np.concatenate((DatOri[:,:bestFeature],DatOri[:,bestFeature+1:]),axis=1) #数据集将bestFeature属性去掉
            subTabel = Table[:]
            subm,subn = np.shape(subDatSet)
            #print subm
            #print "Label:", Label
            if(subm == 0):  #当子集的数据集为空时，说明没有这样的特征样本，根据其父集中样本最多的类
                Tree[bestFeatureTable][value] = majorCnt(Label)#return majorCnt(Label)
            else:
                Tree[bestFeatureTable][value] = TreeGenerate(subDatSet,subDatOri,subTabel)  #Tree[bestFeature][value]两层深度的树

    else:
        for value in [-1,1]:
            if value == -1:
                subDatSetR = Dat[Dat[:,bestFeature] < bestPoint] #选出属性bestFeature，值为value的行
                subDatSet = np.concatenate((subDatSetR[:,:bestFeature],subDatSetR[:,bestFeature+1:]),axis=1) #数据集将bestFeature属性去掉
                subDatOri = np.concatenate((DatOri[:,:bestFeature],DatOri[:,bestFeature+1:]),axis=1) #数据集将bestFeature属性去掉
                subTabel = Table[:]
                subm,subn = np.shape(subDatSet)
                strval = '<' + str(bestPoint)
                if(subm == 0):  #当子集的数据集为空时，说明没有这样的特征样本，根据其父集中样本最多的类
                	Tree[bestFeatureTable][strval] = majorCnt(Label)#return majorCnt(Label)
                else:
                    Tree[bestFeatureTable][strval] = TreeGenerate(subDatSet,subDatOri,subTabel)  #Tree[bestFeature][value]两层深度的树
            if value == 1:
                subDatSetR = Dat[Dat[:,bestFeature] >= bestPoint] #选出属性bestFeature，值为value的行
                subDatSet = np.concatenate((subDatSetR[:,:bestFeature],subDatSetR[:,bestFeature+1:]),axis=1) #数据集将bestFeature属性去掉
                subDatOri = np.concatenate((DatOri[:,:bestFeature],DatOri[:,bestFeature+1:]),axis=1) #数据集将bestFeature属性去掉
                subTabel = Table[:]
                subm,subn = np.shape(subDatSet)
                strval = '>=' + str(bestPoint)
                if(subm == 0):  #当子集的数据集为空时，说明没有这样的特征样本，根据其父集中样本最多的类
                	Tree[bestFeatureTable][strval] = majorCnt(Label)#return majorCnt(Label)
                else:
                    Tree[bestFeatureTable][strval] = TreeGenerate(subDatSet,subDatOri,subTabel)  #Tree[bestFeature][value]两层深度的树
               
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

    
#决策树剪枝
def PostPurn(Tree,featureTable,trainData,testData):
    firstkey = Tree.keys()[0]
    subTree = Tree[firstkey]
    Tree3 = {firstkey:{}}
    #firstselectfeature = featureTable.index(firstkey) #根据属性得到其对应的索引号
    #sub1testData = testData[:,firstselectfeature]
    #subt1rainData = trainData[:,firstselectfeature]
    #print sub1testData
    #print subt1rainData
    for key in subTree.keys():
        #print "key= ",(firstkey,key)
        selectfeature = featureTable.index(firstkey) #根据属性得到其对应的索引号
        subtestData = testData[testData[:,selectfeature] == key]
        subtrainData = trainData[trainData[:,selectfeature] == key]
        subtrainLabel = subtrainData[:,-1]
        sub2Tree = subTree[key]
        #print 'subTree[key]: ',subTree[key]
        if type(subTree[key]).__name__ == 'int':
            Tree3[firstkey][key] = subTree[key]
        else:
            if isendTree(subTree[key]):  # and isTree(sub2Tree):
                Tree2 = subTree.copy()
                #print 'subtrainLabel',subtrainLabel
                Tree2[key] = majorCnt(subtrainLabel) #结果为1个值
                #print Tree2[key]
                Accurateafter = AccurateCalcnotTree(Tree2[key],featureTable,subtestData)
                Accuratebefore = AccurateCalc(Tree,featureTable,subtestData)
                #print Tree2[key]
                #print subTree
                #print subtestData
                #print Accuratebefore,Accurateafter
                if Accurateafter > Accuratebefore:
                    #subTree[key] = Tree2[key]
                    #return Tree2[key]
                    Tree3[firstkey][key] = Tree2[key]
                else: 
                    Tree3[firstkey][key] = subTree[key]
                    #return subTree[key]
                    #subTree[key] = Tree
                    #return subTree[key]
                #return Tree #subTree[key]
            else: Tree3[firstkey][key] = PostPurn(sub2Tree,featureTable,subtrainData,subtestData)
        #print "Tree3: ",Tree3
    return Tree3

def AccurateCalc(Tree,featureTable,testData): #testData为np.array格式，树计算
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

def AccurateCalcnotTree(Tree,featureTable,testData): #testData为np.array格式,叶节点计算
    testDat = testData[:,:-1]
    testLable = testData[:,-1]
    m,n = np.shape(testDat)
    rightcounter = 0
    for num in range(m):
        if(Tree == testLable[num]):
            rightcounter += 1
            #print rightcounter
    return rightcounter/(m*1.0)
    
def isTree(obj):
    return (type(obj).__name__ == 'dict')
    
def isendTree(Tree): #判断是否是节点树,Tree为一个字典
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