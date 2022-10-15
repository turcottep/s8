"""
Classe "ImageCollection" statique pour charger et visualiser les images de la problématique
Membres statiques:
    image_folder: le sous-répertoire d'où les images sont chargées
    image_list: une énumération de tous les fichiers .jpg dans le répertoire ci-dessus
    images: une matrice de toutes les images, (optionnelle, décommenter le code)
    all_images_loaded: un flag qui indique si la matrice ci-dessus contient les images ou non
Méthodes statiques: TODO JB move to helpers
    images_display: affiche quelques images identifiées en argument
    view_histogrammes: affiche les histogrammes de couleur de qq images identifiées en argument
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from skimage import color as skic
from skimage import io as skiio


class ImageCollection:
    """
    Classe globale pour regrouper les infos utiles et les méthodes de la collection d'images
    """

    # liste de toutes les images
    image_folder = r"." + os.sep + "baseDeDonneesImages"
    _path = glob.glob(image_folder + os.sep + r"*.jpg")
    image_list = os.listdir(image_folder)
    # Filtrer pour juste garder les images
    image_list = [i for i in image_list if '.jpg' in i]

    all_images_loaded = False
    images = []

    # # Créer un array qui contient toutes les images
    # # Dimensions [980, 256, 256, 3]
    # #            [Nombre image, hauteur, largeur, RGB]
    # # TODO décommenter si voulu pour charger TOUTES les images
    # images = np.array([np.array(skiio.imread(image)) for image in _path])
    # all_images_loaded = True

    def images_display(indexes):
        """
        fonction pour afficher les images correspondant aux indices
        indexes: indices de la liste d'image (int ou list of int)
        """

        # Pour qu'on puisse traiter 1 seule image
        if type(indexes) == int:
            indexes = [indexes]

        fig2 = plt.figure()
        ax2 = fig2.subplots(len(indexes), 1)

        for i in range(len(indexes)):
            if ImageCollection.all_images_loaded:
                im = ImageCollection.images[i]
            else:
                im = skiio.imread(ImageCollection.image_folder + os.sep + ImageCollection.image_list[indexes[i]])
            ax2[i].imshow(im)

    def view_histogrammes(indexes):
        """
        Affiche les histogrammes de couleur de quelques images
        indexes: int or list of int des images à afficher
        """

        # helper function pour rescaler le format lab
        def rescaleHistLab(LabImage, n_bins):
            """
            Helper function
            La représentation Lab requiert un rescaling avant d'histogrammer parce que ce sont des floats!
            """
            # Constantes de la représentation Lab
            class LabCte:      # TODO JB : utiliser an.Extent?
                min_L: int = 0
                max_L: int = 100
                min_ab: int = -110
                max_ab: int = 110
            # Création d'une image vide
            imageLabRescale = np.zeros(LabImage.shape)
            # Quantification de L en n_bins niveaux
            imageLabRescale[:, :, 0] = np.round(
                (LabImage[:, :, 0] - LabCte.min_L) * (n_bins - 1) / (
                        LabCte.max_L - LabCte.min_L))  # L has all values between 0 and 100
            # Quantification de a et b en n_bins niveaux
            imageLabRescale[:, :, 1:2] = np.round(
                (LabImage[:, :, 1:2] - LabCte.min_ab) * (n_bins - 1) / (
                        LabCte.max_ab - LabCte.min_ab))  # a and b have all values between -110 and 110
            return imageLabRescale


        ###########################################
        # view_histogrammes starts here
        ###########################################
        # TODO JB split calculs et view en 2 fonctions séparées
        # Pour qu'on puisse traiter 1 seule image
        if type(indexes) == int:
            indexes = [indexes]

        fig = plt.figure()
        ax = fig.subplots(len(indexes), 2)

        for num_images in range(len(indexes)):
            # charge une image si nécessaire
            if ImageCollection.all_images_loaded:
                imageRGB = ImageCollection.images[num_images]
            else:
                imageRGB = skiio.imread(
                    ImageCollection.image_folder + os.sep + ImageCollection.image_list[indexes[num_images]])

            # Exemple de conversion de format pour Lab et HSV
            imageLab = skic.rgb2lab(imageRGB)  # TODO L1.E3.5: afficher ces nouveaux histogrammes
            imageHSV = skic.rgb2hsv(imageRGB)  # TODO problématique: essayer d'autres espaces de couleur

            # Number of bins per color channel pour les histogrammes (et donc la quantification de niveau autres formats)
            n_bins = 256

            # Lab et HSV requiert un rescaling avant d'histogrammer parce que ce sont des floats au départ!
            imageLabhist = rescaleHistLab(imageLab, n_bins) # External rescale pour Lab
            imageHSVhist = np.round(imageHSV * (n_bins - 1))  # HSV has all values between 0 and 100

            # Construction des histogrammes
            # 1 histogram per color channel
            pixel_valuesRGB = np.zeros((3, n_bins))
            pixel_valuesLab = np.zeros((3, n_bins))
            pixel_valuesHSV = np.zeros((3, n_bins))
            for i in range(n_bins):
                for j in range(3):
                    pixel_valuesRGB[j, i] = np.count_nonzero(imageRGB[:, :, j] == i)
                    pixel_valuesLab[j, i] = np.count_nonzero(imageLabhist[:, :, j] == i)
                    pixel_valuesHSV[j, i] = np.count_nonzero(imageHSVhist[:, :, j] == i)

            # permet d'omettre les bins très sombres et très saturées aux bouts des histogrammes
            skip = 5
            start = skip
            end = n_bins - skip

            # affichage des histogrammes
            ax[num_images, 0].scatter(range(start, end), pixel_valuesRGB[0, start:end], c='red')
            ax[num_images, 0].scatter(range(start, end), pixel_valuesRGB[1, start:end], c='green')
            ax[num_images, 0].scatter(range(start, end), pixel_valuesRGB[2, start:end], c='blue')
            ax[num_images, 0].set(xlabel='pixels', ylabel='compte par valeur d\'intensité')
            # ajouter le titre de la photo observée dans le titre de l'histogramme
            image_name = ImageCollection.image_list[indexes[num_images]]
            ax[num_images, 0].set_title(f'histogramme RGB de {image_name}')

            # 2e histogramme
            # TODO L1.E3 afficher les autres histogrammes de Lab ou HSV dans la 2e colonne de subplots
