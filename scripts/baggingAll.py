# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 16:01:15 2024

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
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import mean_squared_error, cohen_kappa_score, classification_report, confusion_matrix




# Exemple de données avec une variable qualitative
data=pd.read_excel("C:\\Users\\philippe\\Desktop\\TestExcel\\datasetAllFirstSecondT.xlsx")
df=pd.DataFrame(data)
dffiltre=df[(df['PourcentageG1']>=30)&(df['PourcentageEx']>=50)]
print(dffiltre)
df1=dffiltre[['Age','EcoledeProvenance','SectionH','Faculte','PourcentageEx','Faculte','Decision']]
#df1=df[['PourcentageEx','PourcentageG1']]
# Encoder la variable qualitative
df_encoded = pd.get_dummies(df1, columns=['EcoledeProvenance','SectionH','Faculte'], drop_first=True)

# Séparer les caractéristiques et la cible
X = df_encoded.drop('Decision', axis=1)
y = df_encoded['Decision']
#X = df1['PourcentageEx']
#y = df1['PourcentageG1']
# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Normaliser les données
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)
basemodel=GaussianNB()
bagging_model=BaggingClassifier(basemodel,n_estimators=200,random_state=42)
#model=LinearRegression()
#model.fit(X_train,y_train)

bagging_model.fit(X_train,y_train)

#y_pred=model.predict(X_test)
y_pred=bagging_model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f'Précision du modèle : {accuracy:.2f}')

# Visualiser l'arbre de décision



print(cohen_kappa_score(y_pred, y_test))



#print(f'Accuracy: {accuracy:.2f}')
print("Matrice de confusion:")
print(confusion_matrix(y_pred, y_test))

print("\nRapport de classification:")
print(classification_report(y_pred, y_test))