# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 10:47:56 2024

@author: philippe
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, cohen_kappa_score, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB

#data=pd.read_excel("C:\\Users\\philippe\\Desktop\\TestExcel\\datasetAllFirstSecond.xlsx")
data=pd.read_excel("C:\\Users\\philippe\\Desktop\\TestExcel\\listecategorie.xlsx")

df=pd.DataFrame(data)
#dffiltre=df[(df['PourcentageG1']>=60)&(df['PourcentageEx']>=50)&(df['Age']>=15)]

# Séparer les variables d'entrée et la variable cible
X = df.drop('Decision', axis=1)
y = df['Decision']

# Définir les colonnes catégorielles et continues
categorical_features = ['Sexe','EcoledeProvenance','SectionH','Faculte']
continuous_features = ['Age', 'PourcentageEx']

# Préparation du préprocesseur
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),
        ('num', 'passthrough', continuous_features)
    ]
)

# Créer un pipeline avec préprocessing et modèle
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entraîner le modèle
pipeline.fit(X_train, y_train)
y_pred=pipeline.predict(X_test)
# Évaluer le modèle
accuracy = pipeline.score(X_test, y_test)
print(cohen_kappa_score(y_test, y_pred))

print(f'Accuracy: {accuracy:.2f}')
print("Matrice de confusion:")
print(confusion_matrix(y_test, y_pred))

print("\nRapport de classification:")
print(classification_report(y_test, y_pred))