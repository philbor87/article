# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 19:39:34 2024

@author: philippe
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, cohen_kappa_score, classification_report, confusion_matrix


# Exemple de données avec une variable qualitative
data=pd.read_excel("C:\\Users\\philippe\\Desktop\\TestExcel\\datasetAllFirstSecondT.xlsx")
df=pd.DataFrame(data)
dffiltre=df[(df['PourcentageG1']>=30)&(df['PourcentageEx']>=50)]
df1=dffiltre[['EcoledeProvenance', 'Sexe','Age','SectionH','Faculte','PourcentageEx','Decision']]

# Encoder la variable qualitative
df_encoded = pd.get_dummies(df1, columns=['EcoledeProvenance','Faculte','Sexe', 'SectionH'], drop_first=True)

# Séparer les caractéristiques et la cible
X = df_encoded.drop('Decision', axis=1)
y = df_encoded['Decision']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Créer et entraîner l'arbre de décision pour la régression
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

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

# Visualiser l'arbre de décision



print(cohen_kappa_score(y_pred, y_test))



#print(f'Accuracy: {accuracy:.2f}')
print("Matrice de confusion:")
print(confusion_matrix(y_pred, y_test))

print("\nRapport de classification:")
print(classification_report(y_pred, y_test))