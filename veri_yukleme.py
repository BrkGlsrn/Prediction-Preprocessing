# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 20:16:22 2018

@author: PackardBell
"""
#Library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#veri yükleme
veriler = pd.read_csv("veriler.csv")

print(veriler)

#veri önişleme

boy = veriler[['boy']]
print(boy)

boykilo= veriler[['boy','kilo']]
print(boykilo)

class insan:
    boy=180
    def kosmak(self,b):
        return b+10


ali = insan()

print(ali.kosmak(20))
