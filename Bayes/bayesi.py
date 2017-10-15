# coding=utf-8

import numpy as np
import pandas as pd
import math

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
    
def loadDatapd(filename):
    data = pd.read_csv(filename,encoding='gb2312')
    del data[u'编号']
    return data
    
def toNum(data_input): #pd格式的中文
    data = data_input.copy()
    map_dictionary = {
                      u'浅白': 0,
                      u'青绿': 1,
                      u'乌黑': 2,
                      u'蜷缩': 0,
                      u'稍蜷': 1,
                      u'硬挺': 2,
                      u'沉闷': 0,
                      u'浊响': 1,
                      u'清脆': 2,
                      u'模糊': 0,
                      u'稍糊': 1,
                      u'清晰': 2,
                      u'凹陷': 0,
                      u'稍凹': 1,
                      u'平坦': 2,
                      u'硬滑': 0,
                      u'软粘': 1,
                      u'是': 1,
                      u'否': 0
    }
    for feat in data.columns[:-4]:
        data[feat] = data_input[feat].map(map_dictionary)
    data[u'好瓜'] = data_input[u'好瓜'].map(map_dictionary)
    '''
    data[u'色泽'] = data_input[u'色泽'].map(map_dictionary)
    data[u'根蒂'] = data_input[u'根蒂'].map(map_dictionary)
    data[u'敲声'] = data_input[u'敲声'].map(map_dictionary)
    data[u'纹理'] = data_input[u'纹理'].map(map_dictionary)
    data[u'脐部'] = data_input[u'脐部'].map(map_dictionary)
    data[u'触感'] = data_input[u'触感'].map(map_dictionary)
    data[u'好瓜'] = data_input[u'好瓜'].map(map_dictionary)
    '''
    return data
    
#P(c|x) = P(c)*P(x|c)/P(x)
def bayes(data_input):
    dataset = data_input.copy()
    Pc = {}   #用于存储P(c)
    Pcx = {}  #用于存储P(x|c)
    Uc = {}
    Ue = {}
    c_valuecount = dataset[u'好瓜'].value_counts() #样本的总数
    for c in c_valuecount.index:
        c_count = c_valuecount[c]
        N_count = dataset[u'好瓜'].count()#sum(axis=0)
        Pc[c] = c_count*1.0/N_count
        data_c = dataset[dataset[u'好瓜']==c]
        
        #区分连续属性和离散属性
        #离散属性
        Pcx[c] = {}
        for x in dataset.columns[:-4]:
            Pcx[c][x] = {}
            c_x_xi_count = data_c[x].value_counts()
            for xi in c_x_xi_count.index:
                c_xi_count = c_x_xi_count[xi]  #[c].value_counts()[xi]
                Pcx[c][x][xi] = c_xi_count*1.0/c_count
                
        #连续属性
        #假设密度和含糖度服从正态分布
        #计算uci

        Uc[c] = {}
        Ue[c] = {}
        for x in dataset.columns[-4:-2]:
            c_x_sum = data_c[x].sum()
            uci = round(c_x_sum,3)*1.0/c_count
            xsubuci2 = ((data_c[x] - uci)**2).cumsum().iloc[c_count-1]
            eci = round(math.sqrt(round(xsubuci2,3)*1.0/c_count),3)
            Uc[c][x] = uci
            Ue[c][x] = eci
        #for xi in dataset[x].values():
        #    Pcx[c][x][xi] = st_norm(xi,uci,eci)
    return Pc,Pcx,(Uc,Ue)
    
def predict(data_input,Pc,Pcx,(Uc,Ue)): #为含表头的DataFrame格式待预测数据
    data = data_input.copy()
    max_probil = 0
    c = -1
    for key in Pc.keys():
        probtmp = Pc[key]
        for x in data.columns[:-2]:
            print Pcx[key][x][data[x].values[0]]
            probtmp = probtmp * Pcx[key][x][data[x].values[0]]  #data[x].values[0]从array中取出数值
        for x in data.columns[-2:]:
            uci = Uc[key][x]
            eci = Ue[key][x]
            #Pcx[key][x][data[x].values] = st_norm(data[x].values,uci,eci)
            probtmp = probtmp * st_norm(data[x].values[0],uci,eci)    #Pcx[key][x][data[x].values]
        prob = probtmp
        print 'the probablity of label %d is %8.7f' %(key,prob)
        if prob > max_probil:
            max_probil = prob
            c = key
    return c
        
        
    
def st_norm(x,uci,eci):
    return 1.0/(math.sqrt(2*math.pi)*eci)*math.e**(-(x - uci)**2/(2*eci**2))