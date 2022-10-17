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
from skimage.restoration import estimate_sigma as skimage_estimate_sigma


class ImageCollection:
    """
    Classe globale pour regrouper les infos utiles et les méthodes de la collection d'images
    """

    # liste de toutes les images
    image_folder = r"." + os.sep + "baseDeDonneesImages"
    _path = glob.glob(image_folder + os.sep + r"*.jpg")
    image_list = os.listdir(image_folder)
    # Filtrer pour juste garder les images
    image_list = [i for i in image_list if ".jpg" in i]

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
                im = skiio.imread(
                    ImageCollection.image_folder
                    + os.sep
                    + ImageCollection.image_list[indexes[i]]
                )
            ax2[i].imshow(im)
            ax2[i].set_title(ImageCollection.image_list[indexes[i]])

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
            class LabCte:  # TODO JB : utiliser an.Extent?
                min_L: int = 0
                max_L: int = 100
                min_ab: int = -110
                max_ab: int = 110

            # Création d'une image vide
            imageLabRescale = np.zeros(LabImage.shape)
            # Quantification de L en n_bins niveaux
            imageLabRescale[:, :, 0] = np.round(
                (LabImage[:, :, 0] - LabCte.min_L)
                * (n_bins - 1)
                / (LabCte.max_L - LabCte.min_L)
            )  # L has all values between 0 and 100
            # Quantification de a et b en n_bins niveaux
            imageLabRescale[:, :, 1:2] = np.round(
                (LabImage[:, :, 1:2] - LabCte.min_ab)
                * (n_bins - 1)
                / (LabCte.max_ab - LabCte.min_ab)
            )  # a and b have all values between -110 and 110
            return imageLabRescale

        ###########################################
        # view_histogrammes starts here
        ###########################################
        # TODO JB split calculs et view en 2 fonctions séparées
        # Pour qu'on puisse traiter 1 seule image
        if type(indexes) == int:
            indexes = [indexes]

        fig = plt.figure()
        ax = fig.subplots(len(indexes), 3)

        for num_images in range(len(indexes)):
            # charge une image si nécessaire
            if ImageCollection.all_images_loaded:
                imageRGB = ImageCollection.images[num_images]
            else:
                imageRGB = skiio.imread(
                    ImageCollection.image_folder
                    + os.sep
                    + ImageCollection.image_list[indexes[num_images]]
                )

            # Exemple de conversion de format pour Lab et HSV
            imageLab = skic.rgb2lab(
                imageRGB
            )  # TODO L1.E3.5: afficher ces nouveaux histogrammes
            imageHSV = skic.rgb2hsv(
                imageRGB
            )  # TODO problématique: essayer d'autres espaces de couleur

            # Number of bins per color channel pour les histogrammes (et donc la quantification de niveau autres formats)
            n_bins = 256

            # Lab et HSV requiert un rescaling avant d'histogrammer parce que ce sont des floats au départ!
            imageLabhist = rescaleHistLab(imageLab, n_bins)  # External rescale pour Lab
            imageHSVhist = np.round(
                imageHSV * (n_bins - 1)
            )  # HSV has all values between 0 and 100

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
            ax[num_images, 0].plot(
                range(start, end), pixel_valuesRGB[0, start:end], c="red"
            )
            ax[num_images, 0].plot(
                range(start, end), pixel_valuesRGB[1, start:end], c="green"
            )
            ax[num_images, 0].plot(
                range(start, end), pixel_valuesRGB[2, start:end], c="blue"
            )
            ax[num_images, 0].set(
                xlabel="pixels", ylabel="compte par valeur d'intensité"
            )
            # ajouter le titre de la photo observée dans le titre de l'histogramme
            image_name = ImageCollection.image_list[indexes[num_images]]
            ax[num_images, 0].set_title(f"histogramme RGB de {image_name}")

            # 2e histogramme
            # TODO L1.E3 afficher les autres histogrammes de Lab ou HSV dans la 2e colonne de subplots
            ax[num_images, 1].plot(
                range(start, end), pixel_valuesLab[0, start:end], c="red"
            )
            ax[num_images, 1].plot(
                range(start, end), pixel_valuesLab[1, start:end], c="green"
            )
            ax[num_images, 1].plot(
                range(start, end), pixel_valuesLab[2, start:end], c="blue"
            )
            ax[num_images, 1].set(
                xlabel="pixels", ylabel="compte par valeur d'intensité"
            )
            # ajouter le titre de la photo observée dans le titre de l'histogramme
            image_name = ImageCollection.image_list[indexes[num_images]]
            ax[num_images, 1].set_title(f"histogramme Lab de {image_name}")

            # 3e histogramme
            # TODO L1.E3 afficher les autres histogrammes de Lab ou HSV dans la 3e colonne de subplots
            ax[num_images, 2].plot(
                range(start, end), pixel_valuesHSV[0, start:end], c="red"
            )
            ax[num_images, 2].plot(
                range(start, end), pixel_valuesHSV[1, start:end], c="green"
            )
            ax[num_images, 2].plot(
                range(start, end), pixel_valuesHSV[2, start:end], c="blue"
            )
            ax[num_images, 2].set(
                xlabel="pixels", ylabel="compte par valeur d'intensité"
            )
            # ajouter le titre de la photo observée dans le titre de l'histogramme
            image_name = ImageCollection.image_list[indexes[num_images]]
            ax[num_images, 2].set_title(f"histogramme HSV de {image_name}")

    def get_noise_levels(indexes):
        """Returns a list of the noise levels of the images in the collection.
        The noise level is the standard deviation of the noise in the image.
        The noise is the difference between the image and its median filtered version.
        """

        noise_levels = []

        for num_images in range(len(indexes)):
            # charge une image si nécessaire
            if ImageCollection.all_images_loaded:
                imageRGB = ImageCollection.images[num_images]
            else:
                imageRGB = skiio.imread(
                    ImageCollection.image_folder
                    + os.sep
                    + ImageCollection.image_list[indexes[num_images]]
                )

            noise_level = skimage_estimate_sigma(
                imageRGB, multichannel=True, average_sigmas=True
            )
            noise_levels.append(noise_level)

        for num_images in range(
            len(indexes)
        ):  # TODO L1.E3.6: afficher les niveaux de bruit dans la console
            print(
                "Niveau de bruit de l'image",
                ImageCollection.image_list[indexes[num_images]],
                "est",
                noise_levels[num_images],
            )
        # print("Noise levels:", noise_levels)
        return noise_levels

    def get_color_values(indexes, r, g, b):
        # returns how correlated the color is with the images
        # r,g,b are the values of the color we want to check
        # indexes are the indexes of the images we want to check
        # returns a list of the correlation values
        # the correlation value is the sum of the absolute value of the difference between the color and the image
        # the lower the value, the more correlated the color is with the image
        # the higher the value, the less correlated the color is with the image

        color_values = []

        for num_images in range(len(indexes)):
            # charge une image si nécessaire
            if ImageCollection.all_images_loaded:
                imageRGB = ImageCollection.images[num_images]
            else:
                imageRGB = skiio.imread(
                    ImageCollection.image_folder
                    + os.sep
                    + ImageCollection.image_list[indexes[num_images]]
                )

            imageHSV = skic.rgb2hsv(imageRGB)

            # TODO L1.E3.7: calculer la valeur de couleur de l'image
            # TODO L1.E3.8: ajouter la valeur de couleur calculée à la liste color_values
            # TODO L1.E3.9: afficher les valeurs de couleur dans la console
            # TODO L1.E3.10: afficher les valeurs de couleur dans un graphique

            # calcul de la valeur de couleur de l'image
            # goal_hsv = skic.rgb2hsv(np.array([[[r, g, b]]]))[0][0]

            goal_hue = skic.rgb2hsv(np.array([[[r, g, b]]]))[0][0][0]
            # # print("goal_hue", goal_hue)
            # lower_bound = np.array([goal_hue - 0.1, 0.2, 0.2])
            # upper_bound = np.array([goal_hue + 0.1, 1, 1])

            lower_bound_white = np.array([0, 0, 0.9])
            upper_bound_white = np.array([1, 0.1, 1])

            lower_bound_grey = np.array([0, 0, 0.5])
            upper_bound_grey = np.array([1, 0.1, 0.9])

            lower_bound = lower_bound_grey
            upper_bound = upper_bound_grey

            nb_pixels_in_range = 0
            nb_pixels_total = 0
            for i in range(imageHSV.shape[0]):
                for j in range(imageHSV.shape[1]):
                    nb_pixels_total += 1
                    if (
                        imageHSV[i, j, 0] > lower_bound[0]
                        and imageHSV[i, j, 0] < upper_bound[0]
                        and imageHSV[i, j, 1] > lower_bound[1]
                        and imageHSV[i, j, 1] < upper_bound[1]
                        and imageHSV[i, j, 2] > lower_bound[2]
                        and imageHSV[i, j, 2] < upper_bound[2]
                    ):
                        nb_pixels_in_range += 1

            color_value = nb_pixels_in_range / nb_pixels_total
            color_values.append(color_value)

            # affichage des valeurs de couleur dans la console
            print(
                num_images,
                "/",
                len(indexes),
                "Valeur de couleur de l'image",
                ImageCollection.image_list[indexes[num_images]],
                "est",
                color_values[num_images],
                end="\r",
            )

        # affichage des valeurs de couleur dans un graphique
        fig, ax = plt.subplots()
        ax.bar(range(len(indexes)), color_values)
        ax.set(
            xlabel="images",
            ylabel="valeur de couleur",
            title="Valeur de couleur pour chaque image",
        )
        # ajouter le titre de la photo observée pour chaque barre
        for num_images in range(len(indexes)):
            image_name = ImageCollection.image_list[indexes[num_images]]
            ax.text(
                num_images,
                color_values[num_images],
                image_name,
                ha="center",
                va="bottom",
            )

        return color_values

    def check_discrimination():
        """Checks if the images in the collection are discriminable.
        The images are considered discriminable if the noise level is different
        between the images.
        """

        # TODO L1.E3.7: afficher un message dans la console si les images sont discriminables
        # ou pas discriminables
        range_max = len(ImageCollection.image_list)
        # metrics = ImageCollection.get_noise_levels(range(range_max))
        metrics = ImageCollection.get_color_values(range(range_max), 255, 255, 0)

        # classify by name on wether the name starts with coast, forest or street
        # and then check if the noise levels are different
        # if they are, the images are discriminable
        # if they are not, the images are not discriminable
        # TODO L1.E3.7: afficher un message dans la console si les images sont discriminables
        # ou pas discriminables

        classes = []
        for i in range(range_max):
            if ImageCollection.image_list[i].startswith("coast"):
                classes.append("coast")
            elif ImageCollection.image_list[i].startswith("forest"):
                classes.append("forest")
            elif ImageCollection.image_list[i].startswith("street"):
                classes.append("street")
            else:
                classes.append("unknown")
        print("classes:", classes)

        # plot the noise levels depending on the class to see if there is clustering
        plt.figure()
        plt.scatter(classes, metrics, alpha=0.1)
        plt.xlabel("classes")
        plt.ylabel("grey levels")
        plt.title("grey levels depending on the class")
