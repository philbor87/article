# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 18:42:00 2025

@author: philippe
"""

"""
Created on Fri Jan 10 23:56:14 2025

@author: philippe
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Étape 1 : Créer un DataFrame avec les choix de cours des étudiants
data=pd.read_excel("C:\\Users\\philippe\\Desktop\\TestExcel\\datasetAllFirstSeconTMult.xlsx")
df=pd.DataFrame(data)

dffiltre=df[(df['PourcentageG1']>=79)&(df['PourcentageG1']<=89)&(df['PourcentageEx']>=50)&(df['Faculte']=='Economie')]
df1=dffiltre[['EcoledeProvenance','Sexe','SectionH','Faculte','PourcentageEx']]
# Appliquer le one-hot encoding
df1['PourcentageEx_binned'] = pd.cut(df1['PourcentageEx'], bins=[50, 60, 70, 80,90], labels=['50-60', '60-70', '70-80','80-90'])

# Créer un DataFrame transactionnel
basket = df1.drop('PourcentageEx', axis=1).copy()
basket = basket.apply(lambda x: x.astype(str))

# Convertir en format "one-hot encoding"
basket_onehot = pd.get_dummies(basket)
# Appliquer l'algorithme Apriori
frequent_itemsets = apriori(basket_onehot, min_support=0.6, use_colnames=True)

# Afficher les itemsets fréquents
pd.set_option('display.max_columns',None)
pd.set_option('display.expand_frame_repr',False)
print("\nItemsets fréquents:")
print(frequent_itemsets)

# Générer des règles d'association
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1,num_itemsets=10)

# Afficher les règles d'association

#print("\nRègles d'association:")
#print(rules)
rules['antecedent_len'] = rules['antecedents'].apply(lambda x: len(x))

# Trier par longueur d'antecedent (plus court est plus général)
rules = rules.sort_values(by='antecedent_len')
hih_confidence_rules=rules[rules['confidence']>0.7]
# Afficher les règles
print("Règles d'association (les plus générales en premier):")
#print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
print(hih_confidence_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])