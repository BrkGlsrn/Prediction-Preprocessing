# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 20:49:00 2018

@author: PackardBell
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


veriler = pd.read_csv("eksikveriler.csv")

print(veriler)

Yas = veriler.iloc[:,1:4].values
print(Yas)

from sklearn.preprocessing import Imputer

imputer= Imputer(missing_values="NaN",strategy="mean",axis=0)

#imputer = imputer.fit(Yas[:,0:3])
#fit ve transformu ayrı ayrı kullanmak yerine aşağıdaki gibi kulanılabilir.
Yas[:,0:3] =imputer.fit_transform(Yas[:,0:3]) 
print(Yas)