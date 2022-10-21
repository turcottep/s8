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

import classifiers
import analysis as an

from cProfile import label
from cgitb import grey
import itertools
from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from analysis import view_classes, Extent
from parameters import (
    get_noise_level,
    get_color_value_from_hsv,
    get_value_from_rgb,
    get_value_from_rgb_shelby,
    get_square_value,
    get_light_pixel_top_image,
)
from skimage import color as skic
from skimage import io as skiio


params_names = [
    "noise_level",
    "green",
    "blue",
    "grey",
    "corners",
    "light_pixel",
    "",
    "",
    "",
    "",
]

class_list = {
    "coast": 0,
    "forest": 1,
    "street": 2,
}

colors = {
    "coast_normal": "blue",
    "coast_beach": "blue",  # "yellow",
    "coast_sunset": "blue",  # "red",
    "forest_green": "green",
    "forest_fall": "green",  # "orange",
    "forest_white": "green",  # "black",
    "street_normal": "orange",  # "gray",
    "unknown": "black",
}


class ImageCollection:
    """
    Classe globale pour regrouper les infos utiles et les méthodes de la collection d'images
    """

    # liste de toutes les images
    image_folder = r"." + os.sep + "images_clean"
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
    images = np.array([np.array(skiio.imread(image)) for image in _path])
    all_images_loaded = True

    def test_square(self):
        """
        Fonction pour tester l'efficacité de la fonction square
        """
        # fig2 = plt.figure()
        # ax2 = fig2.subplots(2, 1)

        # im = skiio.imread(
        #     ImageCollection.image_folder + os.sep + ImageCollection.image_list[500]
        # )
        # ax2[0].imshow(im)
        # ax2[0].set_title(ImageCollection.image_list[0])

        # square_value = get_square_value(im)
        # ax2[1].set_title(square_value)

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
            imageLabRescale[:, :, 1:3] = np.round(
                (LabImage[:, :, 1:3] - LabCte.min_ab)
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

    def get_images_object_list(self):
        range_max = 5  # len(ImageCollection.image_list)

        images_object_list = []
        for i in range(range_max):
            print("Image", i, "/", range_max, end="\r")
            image_name = self.image_list[i]

            main_class_type = image_name.split("_")[0]
            class_type = image_name.split("_")[0] + "_" + image_name.split("_")[1]

            noise_level = get_noise_level(self.images[i])

            green_level = get_color_value_from_hsv(self.images[i], "green")

            blue_level = get_color_value_from_hsv(self.images[i], "blue")

            grey_level = get_color_value_from_hsv(self.images[i], "grey")

            corners = get_square_value(self.images[i])

            light_pixel = get_light_pixel_top_image(self.images[i])

            # self.get_color_value_from_hsv(self.images[i], 0, 255, 0)

            # print(i, "noise_level", param)
            graph_color = colors[class_type]

            params = [
                noise_level,
                green_level,
                blue_level,
                grey_level,
                corners,
                light_pixel,
                0,
                0,
                0,
                0,
            ]

            img_obj = {
                "image": image_name,
                "type": class_type,
                "type_int": class_list[main_class_type],
                "params": params,
                "graph_color": graph_color,
            }
            images_object_list.append(img_obj)

        # print(images_object_list)

        # sort by class
        images_object_list.sort(key=lambda x: x["type"])

    def generate_representation(self):
        images_object_list = self.get_images_object_list()

        data_images = np.zeros(
            (len(images_object_list), len(images_object_list[0]["params"]))
        )

        classes_list = [[] for i in range(len(class_list))]
        print("classes_list", classes_list)

        images_labels = np.zeros(len(images_object_list))

        for i in range(len(images_object_list)):
            data_images[i] = images_object_list[i]["params"]
            images_labels[i] = images_object_list[i]["type_int"]
            classes_list[images_object_list[i]["type_int"]].append(
                images_object_list[i]["params"]
            )

        print("classes_list", classes_list)

        return classes_list, images_labels

    def check_discrimination(self):
        """Checks if the images in the collection are discriminable.
        The images are considered discriminable if the noise level is different
        between the images.
        """
        range_max = 5  # len(ImageCollection.image_list)
        images_object_list = self.get_images_object_list()

        # TODO L1.E3.7: afficher un message dans la console si les images sont discriminables
        # ou pas discriminables
        # metrics = ImageCollection.get_noise_levels(range(range_max))

        # houghLines et sobelxy

        param_index_a = 3
        param_index_b = 4

        param_index = 3

        # get average and standard deviation for each class for noise level
        param_by_class = []
        for type in colors.keys():
            param_by_class.append(
                [
                    x["params"][param_index]
                    for x in images_object_list
                    if x["type"] == type
                ]
            )

        # print("noise_levels", param_by_class)

        # get average and standard deviation for each class for noise level
        param_mean = []
        param_std = []
        for param in param_by_class:
            param_mean.append(np.mean(param))
            param_std.append(np.std(param))

        # print("noise_levels_mean", param_mean)
        # print("noise_levels_std", param_std)

        # metrics = ImageCollection.get_color_values(range(range_max), 255, 0, 0)

        # classify by name on wether the name starts with coast, forest or street
        # and then check if the noise levels are different
        # if they are, the images are discriminable
        # if they are not, the images are not discriminable
        # TODO L1.E3.7: afficher un message dans la console si les images sont discriminables
        # ou pas discriminables

        #### 1 d plot #####

        # # plot the distribution of the metrics for each class
        # plt.figure()

        # for i in range(range_max):
        #     plt.plot(
        #         i,
        #         images_object_list[i]["params"][param_index],
        #         "o",
        #         label=images_object_list[i]["type"],
        #         color=images_object_list[i]["graph_color"],
        #     )

        # # plt.legend()

        # # draw horizontal lines for the mean and standard deviation
        # for i in range(len(colors)):
        #     plt.axhline(
        #         param_mean[i],
        #         color=colors[list(colors.keys())[i]],
        #         linestyle="--",
        #     )
        #     plt.axhline(
        #         param_mean[i] - param_std[i],
        #         color=colors[list(colors.keys())[i]],
        #         linestyle=":",
        #     )
        #     plt.axhline(
        #         param_mean[i] + param_std[i],
        #         color=colors[list(colors.keys())[i]],
        #         linestyle=":",
        #     )

        # # plt.scatter(classes, metrics, alpha=0.1)
        # # plt.xlabel("classes")
        # # plt.ylabel("noise levels")
        # plt.title(params_names[param_index] + " depending on the class")

        #### 2 d plot #####
        plt.figure()

        # generate all possible combinations of parameters
        param_combinations = list(itertools.combinations(range(6), 2))
        print("param_combinations", param_combinations)
        print("param_combinations", len(param_combinations))

        subplot_dim = (4, 4)
        ax, fig = plt.subplots(subplot_dim[0], subplot_dim[1], figsize=(20, 20))
        # fig.tight_layout()

        for i, param_combination in enumerate(param_combinations):
            print("i", i)
            param_index_a = param_combination[0]
            param_index_b = param_combination[1]
            print("param_index_a", param_index_a, "param_index_b", param_index_b)
            plt.subplot(subplot_dim[0], subplot_dim[1], i + 1)

            for j in range(range_max):
                plt.plot(
                    images_object_list[j]["params"][param_index_a],
                    images_object_list[j]["params"][param_index_b],
                    "o",
                    label=images_object_list[j]["type"],
                    color=images_object_list[j]["graph_color"],
                    alpha=0.1,
                )

            # hide the x and y ticks
            plt.xticks([])
            plt.yticks([])

            # hide the x and y values
            plt.tick_params(axis="both", which="both", length=0)

            # add axis labels to the inside of the plot
            plt.xlabel(params_names[param_index_a])
            plt.ylabel(params_names[param_index_b])
            # ax.axes.xaxis.set_label_coords(0.5, -0.1)
            # ax.axes.yaxis.set_label_coords(-0.1, 0.5)

        #### 3 d plot #####
        # fig = plt.figure()

        # ax = fig.add_subplot(111, projection="3d")

        # for i in range(range_max):
        #     ax.scatter(
        #         images_object_list[i]["params"][4],
        #         images_object_list[i]["params"][1],
        #         images_object_list[i]["params"][2],
        #         label=images_object_list[i]["type"],
        #         color=images_object_list[i]["graph_color"],
        #         alpha=0.5,
        #     )


def classify_images(self):
    [data, labels] = self.generate_representation()
    # data = data.reshape(data.shape[0], 1, -1)
    print("data", data)
    print("labels", labels)
    # print("data.shape", data.shape)

    print("labels.shape", labels.shape)
    extent = an.Extent(ptList=data)

    # donneesTest = an.genDonneesTest(ndonnees, TroisClasses.extent)

    # BAYES

    classifiers.full_Bayes_risk(
        data,
        labels,
        data,  # todo: change to test data
        "Bayes risque #1",
        extent,  # todo: change to test extent
        data,  # todo: change to test data
        labels,  #   todo: change to test labels
    )

    #  KNN

    n_clusters = 5  # car les nuages de points sont dispersés dans plusieurs dimensions avec une forme bizarroide, mais on ne veut pas inclure les données aberrantes et on veut garder le temps de calcul raisonnable
    n_neighbors = 3  # car on veut faire la moyenne dans le cas ou on est à la limite de deux classes, puisqu'on a plusieurs points par classe et que certains sont donc plus éloignés du centre que d'autres

    cluster_centers, cluster_labels = classifiers.full_kmean(
        n_clusters,
        data,
        labels,
        "Représentants des 1-moy",
        extent,
    )
    classifiers.full_ppv(
        n_neighbors,
        cluster_centers,
        cluster_labels,
        data,  # todo: change to test data
        "1-PPV sur le 1-moy",
        extent,
        data,
        labels,
    )

    ## NN

    n_hidden_layers = 2  # car c'est suffisant dans la plupart des cas selon la littérature (notes de cours)
    n_neurons = 25  # car on veut être plus gros que le nombre de features en input, mais pas trop pour ne pas sur-apprendre
    learning_rate = 0.01  # standard, à vérifier avec la courbe d'apprentissage
    nb_epochs = 1000  # beaucoup d'époques, mais avec un stop early
    activation_function = "tanh"  # parce que le réseau n'a pas beaucoup de couches cachées et que la fonction tanh est plus efficace que la fonction sigmoïde
    loss_function = "binary_crossentropy"  # parce que c'est une classification binaire pour les différentes classes

    classifiers.full_nn(
        n_hidden_layers,
        n_neurons,
        data,
        labels,
        data,  # todo: change to test data
        f"NN {n_hidden_layers} layer(s) caché(s), {n_neurons} neurones par couche",
        extent,
        data,
        labels,
    )
