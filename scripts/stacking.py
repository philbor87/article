# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 06:29:04 2024

@author: philippe
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, cohen_kappa_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.neural_network import MLPClassifier
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


# Charger les données
data=pd.read_excel("C:\\Users\\philippe\\Desktop\\TestExcel\\datasetAllFirstSeconTbin.xlsx")
df=pd.DataFrame(data)
dffiltre=df[(df['PourcentageG1']>=30)&(df['PourcentageEx']>=50)]
print(dffiltre)
df1=dffiltre[['Age','EcoledeProvenance','Sexe','SectionH','PourcentageEx','Faculte','Decision']]

# Encoder la variable qualitative
df_encoded = pd.get_dummies(df1, columns=['EcoledeProvenance','Sexe','Faculte','SectionH'], drop_first=True)

# Séparer les caractéristiques et la cible
X1 = df_encoded.drop('Decision', axis=1)
y1 = df_encoded['Decision']
X=np.array(X1.values.tolist())
y=np.array(y1.values.tolist())

# Diviser les données en ensembles d'entraînement et de test
X_train1, X_test, y_train1, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train1, y_train1)
print((X_train))
# Modèles de base
#models = [
  
    #LogisticRegression(max_iter=1000),
    #DecisionTreeClassifier(),
    #RandomForestClassifier()
    
#]
models = [
  
   
    DecisionTreeClassifier(random_state=42,max_depth=15,min_samples_leaf=7,criterion='gini'),
    XGBClassifier(
        objective="multi:softmax",
        num_class=2,
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.7,
        colsample_bytree=0.9,
        gamma=1,
        reg_alpha=0,
        reg_lambda=1,
        n_jobs=-1,
        random_state=42,
        num_boost_round=100),
   # LogisticRegression(max_iter=1000)
   MLPClassifier(activation= 'logistic', alpha= 0.001, hidden_layer_sizes= (25,), learning_rate= 'adaptive', solver= 'adam',max_iter=2000,random_state=42)
]

# Créer un tableau pour stocker les prédictions
stacked_predictions_train = np.zeros((X_train.shape[0], len(models)))
stacked_predictions_test = np.zeros((X_test.shape[0], len(models)))

# Validation croisée pour générer des prédictions
kf = KFold(n_splits=5)
for i, model in enumerate(models):
    for train_index, val_index in kf.split(X_train):
        X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
        y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]
       
        model.fit(X_fold_train, y_fold_train)
       
        # Prédictions sur le jeu de validation
        stacked_predictions_train[val_index, i] = model.predict(X_fold_val)
   
    # Prédictions sur l'ensemble de test
    stacked_predictions_test[:, i] = model.predict(X_test)

# Entraîner le modèle de niveau 1
model_final = LogisticRegression()
model_final.fit(stacked_predictions_train, y_train)

# Prédire sur l'ensemble de test avec le modèle final
final_predictions = model_final.predict(stacked_predictions_test)

# Évaluation
print("Accuracy:", accuracy_score(y_test, final_predictions))

accuracy = np.mean(final_predictions == y_test)
print(f'Précision du modèle : {accuracy:.2f}')
print(cohen_kappa_score(final_predictions, y_test))



#print(f'Accuracy: {accuracy:.2f}')
print("Matrice de confusion:")
print(confusion_matrix(final_predictions, y_test))

print("\nRapport de classification:")

print(classification_report(final_predictions, y_test))
# Visualiser l'arbre de décisio
y_pred_proba = model.predict_proba(X_test)

# Afficher les probabilités pour les premières instances
#print("Probabilités des classes pour les premières instances :")
#print(y_pred_proba[:5])
y_scores = model.predict_proba(X_test)[:, 1]  # Probabilités pour la classe positive

# 5. Calculer le TPR et le FPR
fpr, tpr, thresholds = roc_curve(y_test, y_scores)

# 6. Calculer l'aire sous la courbe (AUC)
roc_auc = auc(fpr, tpr)

# 7. Tracer la courbe ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')  # Ligne diagonale
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de faux positifs (FPR)')
plt.ylabel('Taux de vrais positifs (TPR)')
plt.title('Courbe ROC-Stacking')
plt.legend(loc='lower right')
plt.grid()
plt.show()
params=model.get_params()
print(params)
