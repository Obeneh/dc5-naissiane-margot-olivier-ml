import pandas as pd
import unicodedata

# Charger le fichier CSV
df = pd.read_csv('dataset_marketing_grand.csv')

# Nettoyer les noms de colonnes
df.columns = df.columns.str.replace(' ', '_')  # Remplacer les espaces par des underscores

# Fonction pour enlever les accents
def remove_accents(text):
    return ''.join(char for char in unicodedata.normalize('NFD', text) if unicodedata.category(char) != 'Mn')

# Appliquer la fonction pour enlever les accents sur chaque cellule du DataFrame
df = df.map(lambda x: remove_accents(str(x)))

# Enlever les doublons
df.drop_duplicates(inplace=True)

# Sauvegarder le DataFrame nettoyé dans un nouveau fichier CSV
df.to_csv('dataset_marketing_grand_nettoye.csv', index=False)
