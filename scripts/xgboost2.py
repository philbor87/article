# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 13:09:52 2024

@author: philippe
"""

import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Charger le dataset Iris
iris = load_iris()
X, y = iris.data, iris.target

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer un DMatrix à partir des données d'entraînement
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)

# Définir les paramètres du modèle
params = {
    'objective': 'multi:softmax',  # Pour la classification multi-classes
    'num_class': 3,                 # Nombre de classes
    'eta': 0.3,                     # Taux d'apprentissage
    'max_depth': 5,                 # Profondeur maximale des arbres
    'eval_metric': 'mlogloss'       # Métrique d'évaluation
}

# Entraîner le modèle
num_round = 100  # Nombre d'itérations
bst = xgb.train(params, dtrain, num_round)

# Faire des prédictions sur les données de test
y_pred = bst.predict(dtest)

# Évaluer le modèle
accuracy = accuracy_score(y_test, y_pred)
print(f'Précision sur les données de test : {accuracy * 100:.2f}%')
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print("Matrice de confusion :")
print(cm)