# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 18:59:13 2018

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

from sklearn.ensemble import RandomForestRegressor

R_f = RandomForestRegressor(n_estimators=10 , random_state=0)

R_f.fit(X,Y)
tahmin = R_f.predict(X)

print(R_f.predict(6.5))
print(R_f.predict(12))

plt.scatter(X, Y , color ="red")
plt.plot(X,tahmin, color ="navy")


from sklearn.metrics import r2_score

r2 = r2_score(Y,tahmin)













