# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 15:14:48 2024

@author: philippe
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import BaggingRegressor




# Exemple de données avec une variable qualitative
data=pd.read_excel("C:\\Users\\philippe\\Desktop\\TestExcel\\datasetAllFirstSecond.xlsx")
df=pd.DataFrame(data)
dffiltre=df[(df['PourcentageG1']>=40)&(df['PourcentageEx']>=50)]
print(dffiltre)
df1=dffiltre[['Age','EcoledeProvenance','SectionH','PourcentageEx','Faculte','PourcentageG1']]
#df1=df[['PourcentageEx','PourcentageG1']]
# Encoder la variable qualitative
df_encoded = pd.get_dummies(df1, columns=['EcoledeProvenance','SectionH','Faculte'], drop_first=True)

# Séparer les caractéristiques et la cible
X = df_encoded.drop('PourcentageG1', axis=1)
y = df_encoded['PourcentageG1']
#X = df1['PourcentageEx']
#y = df1['PourcentageG1']
# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Normaliser les données
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)
basemodel=MLPRegressor(hidden_layer_sizes=10,max_iter=1000,random_state=42)
bagging_model=BaggingRegressor(base_estimator=basemodel,n_estimators=100,random_state=42)
#model=LinearRegression()
#model.fit(X_train,y_train)

bagging_model.fit(X_train,y_train)

#y_pred=model.predict(X_test)
y_pred=bagging_model.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(f'Mean squared Error:{mse}')
print(f'R2 score:{r2}')