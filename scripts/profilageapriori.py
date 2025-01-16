# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 23:56:14 2025

@author: philippe
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Étape 1 : Créer un DataFrame avec les choix de cours des étudiants
data=pd.read_excel("C:\\Users\\philippe\\Desktop\\TestExcel\\datasetAllFirstSeconTMult.xlsx")
df=pd.DataFrame(data)

dffiltre=df[(df['PourcentageG1']>=70)&(df['PourcentageEx']>=50)&(df['Faculte']=='Info')]
df1=dffiltre[['EcoledeProvenance','Sexe','SectionH','Faculte']]
# Appliquer le one-hot encoding
df_binaire = pd.get_dummies(df1, prefix=['Var1', 'Var2', 'Var3', 'Var4'], drop_first=True)

print("\nDataFrame après One-Hot Encoding:")
print(df_binaire)
# Appliquer l'algorithme Apriori
frequent_itemsets = apriori(df_binaire, min_support=0.2, use_colnames=True)

# Afficher les itemsets fréquents
print("\nItemsets fréquents:")
print(frequent_itemsets)

# Générer des règles d'association
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1,num_itemsets=10)

# Afficher les règles d'association
pd.set_option('display.max_columns',None)
pd.set_option('display.expand_frame_repr',False)
print("\nRègles d'association:")
print(rules)