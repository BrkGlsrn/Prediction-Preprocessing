# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 12:04:10 2018

@author: PackardBell
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


veriler = pd.read_csv("veriler.csv")

ulke = veriler.iloc[:,0:1].values


from sklearn.preprocessing import LabelEncoder 

le = LabelEncoder()
ulke[:,0]=le.fit_transform(ulke[:,0])
print(ulke)

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(categorical_features='all')
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)