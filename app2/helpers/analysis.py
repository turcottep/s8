"""
Fonctions utiles pour le traitement de données
APP2 S8 GIA
Classe disponible:
    Extent: bornes ou plage utile de données

Fonctions disponibles:
    viewEllipse: ajoute une ellipse à 1 sigma sur un graphique
    view_classes: affiche sur un graphique 2D les points de plusieurs classes
    view_classification_results: affichage générique de résultats de classification
    plot_metrics: itère et affiche toutes les métriques d'entraînement d'un RN en regroupant 1 métrique entraînement
                + la même métrique de validation sur le même subplot
    creer_hist2D: crée la densité de probabilité d'une série de points 2D
    view3D: génère un graphique 3D de classes

    calcModeleGaussien: calcule les stats de base d'une série de données
    decorrelate: projette un espace sur une nouvelle base de vecteurs

    genDonneesTest: génère un échantillonnage aléatoire dans une plage 2D spécifiée

    scaleData: borne les min max e.g. des données d'entraînement pour les normaliser
    scaleDataKnownMinMax: normalise des données selon un min max déjà calculé
    descaleData: dénormalise des données selon un min max (utile pour dénormaliser une sortie prédite)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import cm
import itertools
import math


class Extent:
    """
    classe pour contenir les min et max de données 2D
    membres: xmin, xmax, ymin, ymax
    Constructeur peut utiliser les 4 valeurs précédentes ou
        calculer directement les min et max d'une liste de points
    Accesseurs:
        get_array: retourne les min max formattés en array
        get_corners: retourne les coordonnées des points aux coins d'un range couvert par les min max
    """
    def __init__(self, xmin=0, xmax=10, ymin=0, ymax=10, ptList=None):
        """
        Constructeur
        2 options:
            passer 4 arguments min et max
            passer 1 array qui contient les des points sur lesquels sont calculées les min et max
        """
        if ptList is not None:
            self.xmin = np.floor(np.min(ptList[:,0]))-1
            self.xmax = np.ceil(np.max(ptList[:,0]))+1
            self.ymin = np.floor(np.min(ptList[:,1]))-1
            self.ymax = np.ceil(np.max(ptList[:,1]))+1
        else:
            self.xmin = xmin
            self.xmax = xmax
            self.ymin = ymin
            self.ymax = ymax

    def get_array(self):
        """
        Accesseur qui retourne sous format matriciel
        """
        return [[self.xmin, self.xmax], [self.ymin, self.ymax]]

    def get_corners(self):
        """
        Accesseur qui retourne une liste points qui correspondent aux 4 coins d'un range 2D bornés par les min max
        """
        return np.array(list(itertools.product([self.xmin, self.xmax], [self.ymin, self.ymax])))


def viewEllipse(data, ax, scale=1, facecolor='none', edgecolor='red', **kwargs):
    """
    ***Testé seulement sur les données du labo
    Ajoute une ellipse à distance 1 sigma du centre d'une classe
    Inspiration de la documentation de matplotlib 'Plot a confidence ellipse'

    data: données de la classe, les lignes sont des données 2D
    ax: axe des figures matplotlib où ajouter l'ellipse
    scale: Facteur d'échelle de l'ellipse, peut être utilisé comme paramètre pour tracer des ellipses à une
        équiprobabilité différente, 1 = 1 sigma
    facecolor, edgecolor, and kwargs: Arguments pour la fonction plot de matplotlib

    retourne l'objet Ellipse créé
    """
    moy, cov, lambdas, vectors = calcModeleGaussien(data)
    # TODO L2.E1.2 Remplacer les valeurs bidons par les bons paramètres à partir des stats ici
    ellipse = Ellipse((1,1), width=scale, height=scale,
                      angle=0, facecolor=facecolor,
                      edgecolor=edgecolor, linewidth=2, **kwargs)
    return ax.add_patch(ellipse)


def view_classes(data, extent, border_coeffs=None):
    """
    Affichage des classes dans data
    *** Fonctionne pour des classes 2D

    data: tableau des classes à afficher. La première dimension devrait être égale au nombre de classes.
    extent: bornes du graphique
    border_coeffs: coefficient des frontières, format des données voir helpers.classifiers.get_borders()
        coef order: [x**2, xy, y**2, x, y, cst (cote droit log de l'equation de risque), cst (dans les distances de mahalanobis)]
    """
    #  TODO: rendre général, seulement 2D pour l'instant
    dims = np.asarray(data).shape

    fig1, ax1 = plt.subplots(1, 1)
    ax1.set_title(r'Visualisation des classes, des ellipses à distance 1$\sigma$' + ('et des frontières' if border_coeffs is not None else ''))

    #  TODO: rendre général, seulement 3 classes pour l'instant
    colorpoints = ['orange', 'purple', 'black']
    colorfeatures = ['red', 'green', 'blue']

    for i in range(dims[0]):
        tempdata = data[i]
        m, cov, valpr, vectprop = calcModeleGaussien(tempdata)
        ax1.scatter(tempdata[:, 0], tempdata[:, 1], s=5, c=colorpoints[i])
        ax1.scatter(m[0], m[1], c=colorfeatures[i])
        viewEllipse(tempdata, ax1, edgecolor=colorfeatures[i])

    # Ajout des frontières
    if border_coeffs is not None:
        x, y = np.meshgrid(np.linspace(extent.xmin, extent.xmax, 400),
                           np.linspace(extent.ymin, extent.ymax, 400))
        for i in range(math.comb(dims[0], 2)):
            # rappel: coef order: [x**2, xy, y**2, x, y, cst (cote droit log de l'equation de risque), cst (dans les distances de mahalanobis)]
            ax1.contour(x, y,
                        border_coeffs[i][0] * x ** 2 + border_coeffs[i][2] * y ** 2 +
                        border_coeffs[i][3] * x + border_coeffs[i][6] +
                        border_coeffs[i][1] * x * y + border_coeffs[i][4] * y, [border_coeffs[i][5]])

    ax1.set_xlim([extent.xmin, extent.xmax])
    ax1.set_ylim([extent.ymin, extent.ymax])

    ax1.axes.set_aspect('equal')


def view_classification_results(train_data, test1, c1, c2, glob_title, title1, title2, extent, test2=None, c3=None, title3=None):
    """
    Génère 1 graphique avec 3 subplots:
        1. Des données "d'origine" train_data avec leur étiquette encodée dans la couleur c1
        2. Un aperçu de frontière de décision au moyen d'un vecteur de données aléatoires test1 avec leur étiquette
            encodée dans la couleur c2
        3. D'autres données classées test2 (opt) avec affichage encodée dans la couleur c3
    :param train_data:
    :param test1:
    :param test2:
        données à afficher
    :param c1:
    :param c2:
    :param c3:
        couleurs
        c1, c2 et c3 sont traités comme des index dans un colormap
    :param glob_title:
    :param title1:
    :param title2:
    :param title3:
        titres de la figure et des subplots
    :param extent:
        range des données
    :return:
    """
    cmap = cm.get_cmap('seismic')
    if np.asarray(test2).any():
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax3.scatter(test2[:, 0], test2[:, 1], s=5, c=cmap(c3))
        ax3.set_title(title3)
        ax3.set_xlim([extent.xmin, extent.xmax])
        ax3.set_ylim([extent.ymin, extent.ymax])
        ax3.axes.set_aspect('equal')
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle(glob_title)
    ax1.scatter(train_data[:, 0], train_data[:, 1], s=5, c=c1, cmap='viridis')
    ax2.scatter(test1[:, 0], test1[:, 1], s=5, c=c2, cmap='viridis')
    ax1.set_title(title1)
    ax2.set_title(title2)
    ax1.set_xlim([extent.xmin, extent.xmax])
    ax1.set_ylim([extent.ymin, extent.ymax])
    ax2.set_xlim([extent.xmin, extent.xmax])
    ax2.set_ylim([extent.ymin, extent.ymax])
    ax1.axes.set_aspect('equal')
    ax2.axes.set_aspect('equal')


def plot_metrics(model):
    """
    Helper function pour visualiser des métriques d'entraînement de RN
    :param model: réseau de neurones entraîné
    """

    # Détermine le nombre de subplots nécessaires
    i = 0
    for j, metric in enumerate(model.history.history):
        if metric.find('val_') != -1:
            continue
        else:
            i += 1
    [f, axs] = plt.subplots(1, i)

    # remplit les différents subplots
    k = 0
    for j, metric in enumerate(model.history.history):
        # Skip les métriques de validation pour les afficher plus tard
        # sur le même subplot que la même métrique d'entraînement
        if metric.find('val_') != -1:
            continue
        else:
            # Workaround pour subplot() qui veut rien savoir de retourner un array 1D quand on lui demande 1x1
            if i > 1:
                ax = axs[k]
            else:
                ax = axs

            ax.plot([x + 1 for x in model.history.epoch],
                     model.history.history[metric],
                     label=metric)
            if model.history.history.get('val_' + metric):
                ax.plot([x + 1 for x in model.history.epoch],
                         model.history.history['val_' + metric],
                         label='validation ' + metric)
            ax.legend()
            ax.grid()
            ax.set_title(metric)
            k += 1
    f.tight_layout()


def creer_hist2D(data, title, nbin=15, plot=False):
    """
    Crée une densité de probabilité pour une classe 2D au moyen d'un histogramme
    data: liste des points de la classe, 1 point par ligne (dimension 0)

    retourne un array 2D correspondant à l'histogramme
    """

    x = np.array(data[:, 0])
    y = np.array(data[:, 1])

    # TODO L2.E1.1 Faire du pseudocode et implémenter une segmentation en bins...
    # pas des bins de l'histogramme
    deltax = 1
    deltay = 1

    # TODO : remplacer les valeurs bidons par la bonne logique ici
    hist, xedges, yedges = np.histogram2d([1, 1], [1, 1], bins=[1, 1])
    # normalise par la somme (somme de densité de prob = 1)
    histsum = np.sum(hist)
    hist = hist / histsum

    # affichage, commenter l'entièreté de ce qui suit si non désiré
    if plot:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_title(f'Densité de probabilité de {title}')

        # calcule des bords des bins
        xpos, ypos = np.meshgrid(xedges[:-1] + deltax / 2, yedges[:-1] + deltay / 2, indexing="ij")
        dz = hist.ravel()

        # list of colors
        # https://matplotlib.org/stable/tutorials/colors/colormaps.html
        cmap = cm.get_cmap('jet')  # Get desired colormap - you can change this!
        max_height = np.max(dz)  # get range of colorbars so we can normalize
        min_height = np.min(dz)
        # scale each z to [0,1], and get their rgb values
        rgba = [cmap((k - min_height) / max_height) for k in dz]

        ax.bar3d(xpos.ravel(), ypos.ravel(), 0, deltax * .9, deltay * .9, dz, color=rgba)
        # Fin "à commenter" si affichage non désiré

    return hist, xedges, yedges


def view3D(data3D, targets, title):
    """
    Génère un graphique 3D de classes
    :param data: tableau, les 3 colonnes sont les données x, y, z
    :param target: sert à distinguer les classes, expect un encodage one-hot
    """
    colors = np.array([[1.0, 0.0, 0.0],  # Red
                       [0.0, 1.0, 0.0],  # Green
                       [0.0, 0.0, 1.0]])  # Blue
    c = colors[targets]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data3D[:, 0], data3D[:, 1], data3D[:, 2], s=10.0, c=c, marker='x')
    ax.set_title(title)
    ax.set_xlabel('First component')
    ax.set_ylabel('Second component')
    ax.set_zlabel('Third component')
    fig.tight_layout()


def calcModeleGaussien(data, message=''):
    """
    Calcule les stats de base de données
    :param data: les données à traiter, devrait contenir 1 point N-D par ligne
    :param message: si présent, génère un affichage des stats calculées
    :return: la moyenne, la matrice de covariance, les valeurs propres et les vecteurs propres de "data"
    """
    # TODO L1.E2.2 Compléter le code avec les fonctions appropriées ici
    moyenne = [1,2]
    matr_cov = [[2, 1], [1, 2]]
    val_propres, vect_propres = [1, 1], [[2, 1], [1, 2]]
    if message:
        print(message)
        print(f'Moy: {moyenne} \nCov: {matr_cov} \nVal prop: {val_propres} \nVect prop: {vect_propres}')
    return moyenne, matr_cov, val_propres, vect_propres


def decorrelate(data, basis):
    """
    Permet de projeter des données sur une base (pour les décorréler)
    :param data: classes à décorréler, la dimension 0 est le nombre de classes
    :param basis: les vecteurs propres sur lesquels projeter les données
    :return: les données projetées
    """
    dims = np.asarray(data).shape
    decorrelated = np.zeros(np.asarray(data).shape)
    for i in range(dims[0]):
        tempdata = data[i]
        # TODO L1.E2.5 Remplacer l'opération bidon par la bonne projection ici
        decorrelated[i] = tempdata
    return decorrelated


def genDonneesTest(ndonnees, extent):
    # génération de n données aléatoires 2D sur une plage couverte par extent
    return np.transpose(np.array([(extent.xmax - extent.xmin) * np.random.random(ndonnees) + extent.xmin,
                                         (extent.ymax - extent.ymin) * np.random.random(ndonnees) + extent.ymin]))


# usage: OUT = scale_data(IN, MINMAX)
#
# Scale an input vector or matrix so that the values
# are normalized in the range [-1, 1].
#
# Input:
# - IN, the input vector or matrix.
#
# Output:
# - OUT, the scaled input vector or matrix.
# - MINMAX, the original range of IN, used later as scaling parameters.
#
def scaleData(x):
    minmax = (np.min(x), np.max(x))
    y = 2.0 * (x - np.min(x)) / (np.max(x) - np.min(x)) - 1
    return y, minmax


def scaleDataKnownMinMax(x, minmax):
    y = 2.0 * (x - minmax[0]) / (minmax[1] - minmax[0]) - 1
    return y


# usage: OUT = descale_data(IN, MINMAX)
#
# Descale an input vector or matrix so that the values
# are denormalized from the range [-1, 1].
#
# Input:
# - IN, the input vector or matrix.
# - MINMAX, the original range of IN.
#
# Output:
# - OUT, the descaled input vector or matrix.
#
def descaleData(x, minmax):
    y = ((x + 1.0) / 2) * (minmax[1] - minmax[0]) + minmax[0]
    return y


