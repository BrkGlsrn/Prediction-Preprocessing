# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 18:19:46 2018

@author: PackardBell
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv("maaslar.csv")

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X = x.values
Y = y.values

from sklearn.tree import DecisionTreeRegressor

D_t = DecisionTreeRegressor(random_state = 0)
D_t.fit(X,Y)
tahmin=D_t.predict(X)


plt.scatter(X,Y , color ="red")
plt.plot(X,tahmin, color ="navy")

from sklearn.metrics import r2_score

r2 = r2_score(Y,tahmin)

'''
Sonuç olarak ayno değerleri döndürmesinin nedeni Decisin tree de verilen değer 
tree de hangi noktaya geliyorsa onu döndürüyor.Yani her halukarda sonuç Y değerlerinden
farklı bir değer olmayacak.
'''
print(D_t.predict(38))
print(D_t.predict(6.6))