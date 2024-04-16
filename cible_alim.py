import matplotlib.pyplot as plt
import pandas as pd

# Chargement du fichier CSV pour voir son contenu
df = pd.read_csv('./dataset_clients.csv')

# Remplacement des valeurs 'Homme' par 1 et 'Femme' par 2 dans la colonne 'Genre'
df['Genre'] = df['Genre'].replace({'Homme': 1, 'Femme': 2})

# Filtrage des données pour ne conserver que les produits beauté
df_alim = df.loc[df['Type de Produit Prefere'] == 'Alimentaire']

# Calcul du nombre d'hommes et de femmes qui aiment les produits beauté
nb_hommes = df_alim[df_alim['Genre'] == 1].shape[0]
nb_femmes = df_alim[df_alim['Genre'] == 2].shape[0]

# Création des labels et des tailles pour le diagramme circulaire
labels = 'Hommes', 'Femmes'
sizes = [nb_hommes, nb_femmes]
colors = ['#9D5EE6', '#D08AEB']

# Création du diagramme circulaire
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)

# Affichage du diagramme
plt.title('Pie charts - Genre en fonction du type de produit préféré Alimentaire')
plt.show()
