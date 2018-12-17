# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 12:40:59 2018

@author: PackardBell
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv("eksikveriler.csv")
Yas = veriler.iloc[:,1:4].values
ulke = veriler.iloc[:,0:1].values
cinsiyet = veriler.iloc[:,4].values


#Eksik veriler
from sklearn.preprocessing import Imputer

imputer= Imputer(missing_values="NaN",strategy="mean",axis=0)
Yas[:,0:3] =imputer.fit_transform(Yas[:,0:3]) 

#Encoding Nominal veya Ordinal-->Numeric 
from sklearn.preprocessing import LabelEncoder 

le = LabelEncoder()
ulke[:,0]=le.fit_transform(ulke[:,0]) 


from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(categorical_features='all')
ulke = ohe.fit_transform(ulke).toarray()

#data frame oluşturma
sonuc = pd.DataFrame(data = ulke , index = range(22), columns = ['fr','tr','us'])
print(sonuc)

sonuc2 = pd.DataFrame(data = Yas , index = range(22), columns = ['boy','kilo','yas'])

print(sonuc2)

sonuc3 =pd.DataFrame(data = cinsiyet , index = range(22), columns = ['cinsiyet'])

print(sonuc3)

#frame leri birleştirme
s=pd.concat([sonuc,sonuc2],axis=1)
print(s)

#test ve eğitim olarak frame leri bölme
from sklearn.cross_validation import train_test_split

x_train,x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33,random_state=0)


#standartlastırma
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_Train=sc.fit_transform(x_train)







