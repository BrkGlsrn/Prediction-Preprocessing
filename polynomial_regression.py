# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 19:20:57 2018

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
 
#polynomial regression

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(X)
print(x_poly)

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(x_poly,Y)
tahmin = lin_reg.predict(x_poly)


#Plotting 
plt.scatter(X,Y)
plt.plot(X,tahmin,color='red')


#New Predictions
print(lin_reg.predict(poly_reg.fit_transform(11)))
print(lin_reg.predict(poly_reg.fit_transform(6.6)))







