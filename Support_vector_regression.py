# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 11:15:11 2018

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

from sklearn.preprocessing import StandardScaler

sc1=StandardScaler()
x_scaling=sc1.fit_transform(X)
sc2=StandardScaler()
y_scaling=sc1.fit_transform(Y)


from sklearn.svm import SVR 

svr_reg = SVR(kernel='rbf',C=1e3, gamma=0.1)
svr_reg.fit(x_scaling,y_scaling)

tahmin = svr_reg.predict(x_scaling)

plt.scatter(x_scaling,y_scaling , color='navy')
plt.plot(x_scaling,tahmin)

print(svr_reg.predict(sc1.fit_transform(6.6)))

from sklearn.metrics import r2_score

r2 = r2_score(y_scaling,tahmin)








