# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 18:26:36 2018

@author: PackardBell
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv("satislar.csv")

aylar = veriler[['Aylar']]
satislar = veriler[['Satislar']]


from sklearn.cross_validation import train_test_split

x_train,x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33,random_state=0)

#Standartization yapılarakta linear regression uygulunabilir
'''
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_Train=sc.fit_transform(x_train)
Y_Train=sc.fit_transform(y_train)
X_Test=sc.fit_transform(x_test)
Y_Test=sc.fit_transform(y_test)
'''

#Linear Regression
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)

tahmin = lr.predict(x_test)


#plotting

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train,y_train,color="red")
plt.plot(x_test,tahmin)

from sklearn.metrics import r2_score

r2 = r2_score(y_test,tahmin)













