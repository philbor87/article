# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 09:48:31 2024

@author: philippe
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, cohen_kappa_score, classification_report, confusion_matrix,roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier




# Exemple de données avec une variable qualitative
data=pd.read_excel("C:\\Users\\philippe\\Desktop\\TestExcel\\datasetAllFirstSeconTbin.xlsx")
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
# Normaliser les données
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)
#basemodel=DecisionTreeClassifier(max_depth=10)
basemodel=RandomForestClassifier()
ada_model=AdaBoostClassifier(basemodel,n_estimators=100,random_state=42)
#model=LinearRegression()
#model.fit(X_train,y_train)

ada_model.fit(X_train,y_train)

#y_pred=model.predict(X_test)
y_pred=ada_model.predict(X_test)
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


# Afficher un rapport de classification

# Prédire les probabilités
y_pred_proba = ada_model.predict_proba(X_test)

# Afficher les probabilités pour les premières instances
print("Probabilités des classes pour les premières instances :")
print(y_pred_proba[:5])
y_scores =ada_model.predict_proba(X_test)[:, 1]  # Probabilités pour la
y_pred_proba = ada_model.predict_proba(X_test)

# Afficher les probabilités pour les premières instances
print("Probabilités des classes pour les premières instances :")
print(y_pred_proba[:5])
y_scores = ada_model.predict_proba(X_test)[:, 1]  # Probabilités pour la classe positive

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
plt.title('Courbe ROC-Adaboost')
plt.legend(loc='lower right')
plt.grid()
plt.show()
params= ada_model.get_params()
print(params)