"""
Classe "TroisCLasses" statique pour charger les données de 3 classes du laboratoire
Membres statiques:
    C1, C2, C3: les points 2D de chaque classe
    data: les points de toutes les classes à la suite dans 1 seul array
    ndata: le nombre total de points
    class_labels: vecteur des étiquettes de classe pour le vecteur data. 0 = C1, 1 = C2, 2 = C3
    extent: la plage utile d'un graphique pour les données d'origine
    donnees_test: un échantillonage aléatoire de la plage utile
    ndonnees: le nombre total de données de test
"""

import numpy as np
import os
import helpers.analysis as an


class TroisClasses:

    # Import data from text files in subdir
    C1 = np.loadtxt('data_3classes'+os.sep+'C1.txt')
    C2 = np.loadtxt('data_3classes'+os.sep+'C2.txt')
    C3 = np.loadtxt('data_3classes'+os.sep+'C3.txt')

    # reorganisation en 1 seul vecteur pour la suite
    data = np.array([C1, C2, C3])
    _x, _y, _z = data.shape
    # Chaque ligne de data contient 1 point en 2D
    # Les points des 3 classes sont mis à la suite en 1 seul long array
    data = data.reshape(_x * _y, _z)
    ndata = len(data)

    # assignation des classes d'origine 0 à 2 pour C1 à C3 respectivement
    class_labels = np.zeros([ndata, 1])
    class_labels[range(len(C1), 2 * len(C1))] = 1
    class_labels[range(2 * len(C1), ndata)] = 2

    # Min et max des données
    extent = an.Extent(ptList=data)
