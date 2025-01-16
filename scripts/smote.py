# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 17:24:43 2024

@author: philippe
"""
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# 1. Générer un ensemble de données déséquilibré
X, y = make_classification(n_classes=2, class_sep=2,
                           weights=[0.9, 0.1], n_informative=3,
                           n_redundant=1, flip_y=0,
                           n_features=20, n_clusters_per_class=1,
                           n_samples=1000, random_state=42)

# 2. Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Appliquer SMOTE sur l'ensemble d'entraînement
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 4. Afficher la distribution des classes avant et après SMOTE
print("Distribution des classes avant SMOTE:")
print(np.bincount(y_train))

print("Distribution des classes après SMOTE:")
print(np.bincount(y_resampled))

# 5. Entraîner un modèle (Random Forest dans cet exemple)
model = RandomForestClassifier(random_state=42)
model.fit(X_resampled, y_resampled)

# 6. Faire des prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# 7. Évaluer le modèle
print("\nRapport de classification:")
print(classification_report(y_test, y_pred))

print("Matrice de confusion:")
print(confusion_matrix(y_test, y_pred))

# 8. Visualisation (facultatif)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Distribution des classes avant SMOTE")
plt.bar(['Classe 0', 'Classe 1'], np.bincount(y_train), color=['blue', 'orange'])

plt.subplot(1, 2, 2)
plt.title("Distribution des classes après SMOTE")
plt.bar(['Classe 0', 'Classe 1'], np.bincount(y_resampled), color=['blue', 'orange'])

plt.show()