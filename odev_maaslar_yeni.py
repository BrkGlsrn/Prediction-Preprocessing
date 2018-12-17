# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 19:48:11 2018

@author: PackardBell
""" 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
import statsmodels.api as sm

veriler = pd.read_csv("maaslar_yeni.csv")

Unvan_Kidem_Puan = veriler.iloc[:,2:5]
Maas = veriler.iloc[:,5:]

X = Unvan_Kidem_Puan.values
Y = Maas.values

print(veriler.corr())
#-----------------------------------------
'''
from sklearn.cross_validation import train_test_split

x_train,x_test,y_train,y_test = train_test_split(Unvan_Kidem_Puan,Maas,test_size=0.33,random_state=0)
'''
#-----------------------------------------
from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
x_scaling=sc1.fit_transform(X)
sc2=StandardScaler()
y_scaling=sc1.fit_transform(Y)

#-----------------------------------------
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X,Y)
MLR_tahmin = lr.predict(X)
print("Simple Linear Regressin R2 value:")
print(r2_score(Y,MLR_tahmin))
print("CEO MAAŞ TAHMİN:")
print(lr.predict([[10,5,100]]))
print("MUDUR MAAŞ TAHMİN:")
print(lr.predict([[7,5,100]]))
print("------------------------")

model = sm.OLS(lr.predict(X),X)
print(model.fit().summary())

#-----------------------------------------
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=3)
x_poly = poly_reg.fit_transform(X)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x_poly,Y)
PR_tahmin = lin_reg.predict(x_poly)
print("Polynomial Regressin R2 value:")
print(r2_score(Y,PR_tahmin))
print("CEO MAAŞ TAHMİN:")
print(lin_reg.predict(poly_reg.fit_transform([[10,5,100]])))
print("MUDUR MAAŞ TAHMİN:")
print(lin_reg.predict(poly_reg.fit_transform([[7,5,100]])))
print("------------------------")

model2 = sm.OLS(PR_tahmin,X)
print(model2.fit().summary())

#-----------------------------------------
from sklearn.svm import SVR 
svr_reg = SVR(kernel='rbf',C=1e3, gamma=0.1)
svr_reg.fit(x_scaling,y_scaling)
SVR_tahmin = svr_reg.predict(x_scaling)
print("Support Vector Regressin R2 value:")
print(r2_score(y_scaling,SVR_tahmin))
print("CEO MAAŞ TAHMİN:")
print(svr_reg.predict(sc1.fit_transform([[10,5,100]])))
print("MUDUR MAAŞ TAHMİN:")
print(svr_reg.predict(sc1.fit_transform([[7,5,100]])))
print("------------------------")

model3 = sm.OLS(SVR_tahmin,x_scaling)
print(model3.fit().summary())

#-----------------------------------------
from sklearn.tree import DecisionTreeRegressor
D_t = DecisionTreeRegressor(random_state = 0)
D_t.fit(X,Y)
DT_tahmin = D_t.predict(X)
print("Decision Tree Regressin R2 value:")
print(r2_score(Y,DT_tahmin))
print("CEO MAAŞ TAHMİN:")
print(D_t.predict([[10,5,100]]))
print("MUDUR MAAŞ TAHMİN:")
print(D_t.predict([[7,5,100]]))
print("------------------------")

#-----------------------------------------
from sklearn.ensemble import RandomForestRegressor
R_f = RandomForestRegressor(n_estimators=10 , random_state=0)
R_f.fit(X,Y)
RF_tahmin = R_f.predict(X)
print("Random Forest Regressin R2 value:")
print(r2_score(Y,RF_tahmin))
print("CEO MAAŞ TAHMİN:")
print(R_f.predict([[10,5,100]]))
print("MUDUR MAAŞ TAHMİN:")
print(R_f.predict([[7,5,100]]))
print("------------------------")




























