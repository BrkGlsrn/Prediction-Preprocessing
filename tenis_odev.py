# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 19:12:34 2018

@author: PackardBell
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv("odev_tenis.csv")

weather = veriler.iloc[:,0:1].values
temp_humd = veriler.iloc[:,1:3].values
windy = veriler.iloc[:,3:4].values
play = veriler.iloc[:,4:].values


#weather encoding
from sklearn.preprocessing import LabelEncoder 

le = LabelEncoder()
weather[:,0]=le.fit_transform(weather[:,0]) 

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(categorical_features='all')
weather = ohe.fit_transform(weather).toarray()

#windy encoding
from sklearn.preprocessing import LabelEncoder 

le = LabelEncoder()
windy[:,0]=le.fit_transform(windy[:,0]) 

ohe = OneHotEncoder(categorical_features='all')
windy = ohe.fit_transform(windy).toarray()

#play encoding
from sklearn.preprocessing import LabelEncoder 

le = LabelEncoder()
play[:,0]=le.fit_transform(play[:,0]) 

ohe = OneHotEncoder(categorical_features='all')
play = ohe.fit_transform(play).toarray()


#Data frame olusturma
weathers = pd.DataFrame(data = weather , index = range(14), columns = ['Overcast','Rainy','Sunny'])

windys =pd.DataFrame(data = windy[:,-1] , index = range(14), columns = ['Windy'])

temp_humds = pd.DataFrame(data = temp_humd , index = range(14), columns = ['Temperature','Humidity'])

plays =pd.DataFrame(data = play[:,-1] , index = range(14), columns = ['Play'])

last_table=pd.concat([weathers,temp_humds,windys],axis=1)



from sklearn.cross_validation import train_test_split

x_train_2,x_test_2,y_train_2,y_test_2 = train_test_split(last_table,plays,test_size=0.33,random_state=0)


from sklearn.linear_model import LinearRegression
lr2=LinearRegression()
lr2.fit(x_train_2,y_train_2)

tahmin_2 = lr2.predict(x_test_2)
print(tahmin_2)

#Bacward elimination and regression report
import statsmodels.formula.api as sm
#X = np.append(arr=np.ones((22,1)).astype(int), values=last_table_2 , axis=1)
X_L = last_table.iloc[:,[0,1,2,3,4,5]].values
r =sm.OLS(endog = plays , exog =X_L).fit()
print(r.summary())

#En yüksek p değernin değişkeni çıkarıldı
x_train_2 = x_train_2.iloc[:,[0,1,2,4,5]]
x_test_2 = x_test_2.iloc[:,[0,1,2,4,5]]

#yeniden tahmin edildi
lr2.fit(x_train_2,y_train_2)
tahmin_2 = lr2.predict(x_test_2)




