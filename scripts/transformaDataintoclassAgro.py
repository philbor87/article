# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 17:49:08 2024

@author: philippe
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Exemple de données avec une variable qualitative
data=pd.read_excel("C:\\Users\\philippe\\Desktop\\TestExcel\\datasetAllFirstSecondT.xlsx")
df=pd.DataFrame(data)

for i in range(len(df)):
    decision=df.at[i,'Decision']
    if 'GD' in str(decision):
        df.at[i,'Decision'] =3
    elif 'PGD' in str(decision):
        df.at[i,'Decision'] =3
    elif 'D' in str(decision):
        df.at[i,'Decision'] =2
    elif 'S' in str(decision):
        df.at[i,'Decision'] =1
    else:
        df.at[i,'Decision'] =0
print(df)
enregistre=pd.ExcelWriter("C:\\Users\\philippe\\Desktop\\TestExcel\\datasetAllFirstSeconTMult.xlsx")
df.to_excel(enregistre, 'Feuille 1')
enregistre.close()
