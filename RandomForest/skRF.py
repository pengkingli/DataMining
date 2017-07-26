#-*- coding:utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)
features = df.columns[:4] #先选出前四个的columns，然后根据columns选出前n列数据
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75 #在0-1之间产生len(df)个均匀分布的样本，用于选择训练集
df['species'] = iris.target
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
train,test = df[df['is_train']==True],df[df['is_train']==False]

clf = RandomForestClassifier(n_jobs=2) #并行数2
y, _ = pd.factorize(train['species']) #将字符串再次转化为对用数字
clf.fit(train[features],y)

pre = clf.predict(test[features])
actre, _ = pd.factorize(test['species']) 


