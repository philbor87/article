# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 19:48:40 2024

@author: philippe
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, cohen_kappa_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Exemple de données avec une variable qualitative
data=pd.read_excel("C:\\Users\\philippe\\Desktop\\TestExcel\\datasetAllFirstSecondT.xlsx")
df=pd.DataFrame(data)
dffiltre=df[(df['PourcentageG1']>=30)&(df['PourcentageEx']>=50)]
print(dffiltre)
df1=dffiltre[['Age','EcoledeProvenance','Sexe','SectionH','PourcentageEx','Faculte','Decision']]

# Encoder la variable qualitative
df_encoded = pd.get_dummies(df1, columns=['EcoledeProvenance','Sexe','Faculte','SectionH'], drop_first=True)

# Séparer les caractéristiques et la cible
X = df_encoded.drop('Decision', axis=1)
y = df_encoded['Decision']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
# Créer et entraîner l'arbre de décision pour la régression
model=neighbors.KNeighborsClassifier(n_neighbors=5)
#model.fit(X_train,y_train)
model.fit(X_resampled, y_resampled)

#y_pred=model.predict(X_test)
y_pred=model.predict(X_test)
#===========Pour la regression-----------------
# Évaluer le modèle
#y_pred = model.predict(X_test)
#mse = mean_squared_error(y_test, y_pred)
#r2 = r2_score(y_test, y_pred)

#print(f'Erreur quadratique moyenne : {mse:.2f}')
#print(f'Coefficient de détermination R² : {r2:.2f}')

# Visualiser l'arbre de décision
#plt.figure(figsize=(10, 6))
#plot_tree(model, filled=True, feature_names=X.columns)
#plt.title("Arbre de Décision pour la Régression")
#plt.show()
#===== Pour la classification===================
# Évaluer le modèle
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f'Précision du modèle : {accuracy:.2f}')
print(cohen_kappa_score(y_pred, y_test))



#print(f'Accuracy: {accuracy:.2f}')
print("Matrice de confusion:")
print(confusion_matrix(y_pred, y_test))

print("\nRapport de classification:")
print(classification_report(y_pred, y_test))
# Visualiser l'arbre de décision


#plt.figure(figsize=(20, 10))
#plot_tree(model, filled=True, feature_names=X.columns, class_names=['Classe 0', 'Classe 1'])
#plt.title("Arbre de Décision")
#plt.show()
