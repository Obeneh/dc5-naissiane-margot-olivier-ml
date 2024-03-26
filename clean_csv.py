import pandas as pd
import unicodedata

# Charger le fichier CSV
df = pd.read_csv('dataset_clients.csv')

# Nettoyer les noms de colonnes
df.columns = df.columns.str.replace(' ', '_')  # Remplacer les espaces par des underscores

# Fonction pour enlever les accents
def remove_accents(text):
    return ''.join(char for char in unicodedata.normalize('NFD', text) if unicodedata.category(char) != 'Mn')

# Appliquer la fonction pour enlever les accents sur chaque cellule du DataFrame
df = df.map(lambda x: remove_accents(str(x)))
"""
# Convertir les types de texte en 1, 2, 3, 4, 5
def convert_text_to_numeric(text):
    if text == 'texte1':
        return 1
    elif text == 'texte2':
        return 2
    elif text == 'texte3':
        return 3
    elif text == 'texte4':
        return 4
    elif text == 'texte5':
        return 5
    else:
        return text

# Appliquer la fonction de conversion sur les colonnes nécessaires
df['type_de_produit_prefere'] = df['type_de_produit_prefere'].apply(convert_text_to_numeric)

"""
# Enlever les doublons
df.drop_duplicates(inplace=True)

# Sauvegarder le DataFrame nettoyé dans un nouveau fichier CSV
df.to_csv('dataset_clients_nettoye.csv', index=False)
