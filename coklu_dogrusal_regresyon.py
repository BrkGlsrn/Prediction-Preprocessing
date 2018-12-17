# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 22:45:37 2018

@author: PackardBell
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv("veriler.csv")

ulke = veriler.iloc[:,0:1].values
cinsiyet = veriler.iloc[:,-1:].values
Yas = veriler.iloc[:,1:4].values
boy_tahmin = veriler.iloc[:,2:4].values
boy = veriler[['boy']]

#ulke encoding
from sklearn.preprocessing import LabelEncoder 

le = LabelEncoder()
ulke[:,0]=le.fit_transform(ulke[:,0]) 


from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(categorical_features='all')
ulke = ohe.fit_transform(ulke).toarray()

#cinsiyet encoding
from sklearn.preprocessing import LabelEncoder 

le = LabelEncoder()
cinsiyet[:,0]=le.fit_transform(cinsiyet[:,0]) 


from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(categorical_features='all')
cinsiyet = ohe.fit_transform(cinsiyet).toarray()

#Data frame olusturma
ulkeler = pd.DataFrame(data = ulke , index = range(22), columns = ['fr','tr','us'])

cinsiyetler =pd.DataFrame(data = cinsiyet[:,:1] , index = range(22), columns = ['cinsiyet'])

boy_kilo_yas = pd.DataFrame(data = Yas , index = range(22), columns = ['boy','kilo','yas'])

kilo_yas = pd.DataFrame(data = boy_tahmin , index = range(22), columns = ['kilo','yas'])

veri=pd.concat([ulkeler,boy_kilo_yas,cinsiyetler],axis=1)

last_table_2=pd.concat([ulkeler,kilo_yas,cinsiyetler],axis=1)
print(last_table_2)

last_table=pd.concat([ulkeler,boy_kilo_yas],axis=1)
print(last_table)

#verileri ölçekleme
from sklearn.cross_validation import train_test_split

x_train,x_test,y_train,y_test = train_test_split(last_table,cinsiyetler,test_size=0.33,random_state=0)


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)

tahmin = lr.predict(x_test)


from sklearn.cross_validation import train_test_split

x_train_2,x_test_2,y_train_2,y_test_2 = train_test_split(last_table_2,boy,test_size=0.33,random_state=0)


from sklearn.linear_model import LinearRegression
lr2=LinearRegression()
lr2.fit(x_train_2,y_train_2)

tahmin_2 = lr2.predict(x_test_2)


 

#Bacward elimination and regression report
import statsmodels.formula.api as sm
#X = np.append(arr=np.ones((22,1)).astype(int), values=last_table_2 , axis=1)
X_L = last_table_2.iloc[:,[0,1,2,3,4,5]].values
r =sm.OLS(endog = boy , exog =X_L).fit()
print(r.summary())



X_L = last_table_2.iloc[:,[0,1,2,3,5]].values
r =sm.OLS(endog = boy , exog =X_L).fit()
print(r.summary())












