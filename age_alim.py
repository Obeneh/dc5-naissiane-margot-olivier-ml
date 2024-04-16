import pandas as pd
import matplotlib.pyplot as plt

# Chargement du fichier CSV
df = pd.read_csv('./dataset_clients.csv')

# Création des tranches d'âges
bins = [18, 25, 35, 45, 55, 65, 70]
labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
df['Tranche d\'âge'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

# Sélection du produit préféré
df_alim = df.loc[df['Type de Produit Prefere'] == 'Alimentaire']

# Comptage du nombre de clients dans chaque catégorie d'âge
age_counts = df_alim['Tranche d\'âge'].value_counts().reindex(labels)

# Création du diagramme circulaire
plt.figure()
plt.pie(age_counts, labels=age_counts.index, autopct='%1.1f%%')
plt.title(f'Répartition de l\'âge des clients pour le produit Alimentaire')
plt.show()
