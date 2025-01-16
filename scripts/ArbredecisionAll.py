# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 14:59:30 2024

@author: philippe
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, cohen_kappa_score, classification_report, confusion_matrix,roc_curve, auc
from imblearn.over_sampling import SMOTE


# Exemple de données avec une variable qualitative
data=pd.read_excel("C:\\Users\\philippe\\Desktop\\TestExcel\\datasetAllFirstSeconTMult.xlsx")
df=pd.DataFrame(data)
dffiltre=df[(df['PourcentageG1']>=30)&(df['PourcentageEx']>=50)]
df1=dffiltre[['EcoledeProvenance', 'Sexe','Age','Faculte','SectionH','PourcentageEx','Decision']]

# Encoder la variable qualitative
df_encoded = pd.get_dummies(df1, columns=['EcoledeProvenance','Faculte','Sexe', 'SectionH'], drop_first=True)

# Séparer les caractéristiques et la cible
X = df_encoded.drop('Decision', axis=1)
y = df_encoded['Decision']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
# Créer et entraîner l'arbre de décision pour la régression
model = DecisionTreeClassifier(random_state=42,max_depth=10,min_samples_leaf=7,criterion='gini')
#model.fit(X_train, y_train)
model.fit(X_resampled,y_resampled)
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


#plt.figure(figsize=(20, 10))
#plot_tree(model, filled=True, feature_names=X.columns, class_names=["Echec","Reussi"],fontsize=14)

#plt.title("Arbre de Décision",fontsize=10)
#plt.show()
print(cohen_kappa_score(y_pred, y_test))



#print(f'Accuracy: {accuracy:.2f}')
print("Matrice de confusion:")
print(confusion_matrix(y_pred, y_test))

print("\nRapport de classification:")
print(classification_report(y_pred, y_test))
#y_scores = model.predict_proba(X_test)[:, 1]  # Probabilités pour la classe positive

# 5. Calculer le TPR et le FPR
#fpr, tpr, thresholds = roc_curve(y_test, y_scores)

# 6. Calculer l'aire sous la courbe (AUC)
#roc_auc = auc(fpr, tpr)

# 7. Tracer la courbe ROC
#plt.figure(figsize=(8, 6))
#plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
#plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')  # Ligne diagonale
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('Taux de faux positifs (FPR)')
#plt.ylabel('Taux de vrais positifs (TPR)')
#plt.title('Courbe ROC-Arbre_decision')
#plt.legend(loc='lower right')
#plt.grid()
#plt.show()