from skimage.restoration import estimate_sigma as skimage_estimate_sigma
from skimage.feature import corner_fast, corner_peaks
from skimage import color as skic
import numpy as np


def get_noise_level(image_rgb):

    noise_level = skimage_estimate_sigma(
        image_rgb, multichannel=True, average_sigmas=True
    )

    return noise_level


def get_color_value_from_hsv(imageRGB, color: str):

    imageHSV = skic.rgb2hsv(imageRGB)

    # TODO L1.E3.7: calculer la valeur de couleur de l'image
    # TODO L1.E3.8: ajouter la valeur de couleur calculée à la liste color_values
    # TODO L1.E3.9: afficher les valeurs de couleur dans la console
    # TODO L1.E3.10: afficher les valeurs de couleur dans un graphique

    # calcul de la valeur de couleur de l'image
    # goal_hsv = skic.rgb2hsv(np.array([[[r, g, b]]]))[0][0]

    # if color == "red":
    #     lower_bound = np.array([0, 0.5, 0.5])
    #     upper_bound = np.array([0.1, 1, 1])

    if color == "white":
        lower_bound = np.array([0, 0, 0.9])
        upper_bound = np.array([1, 0.1, 1])

    if color == "grey":
        lower_bound = np.array([0, 0, 0.25])
        upper_bound = np.array([1, 0.25, 0.75])

    if color == "blue":
        lower_bound = np.array([180 / 360, 0.2, 0.2])
        upper_bound = np.array([260 / 360, 1, 1])

    if color == "green":
        lower_bound = np.array([60 / 360, 0.2, 0.2])
        upper_bound = np.array([160 / 360, 1, 1])

    nb_pixels_total = np.shape(imageHSV)[0] * np.shape(imageHSV)[1]
    nb_pixels_in_range = np.count_nonzero(
        np.logical_and(
            np.logical_and(
                np.logical_and(
                    (imageHSV[:, :, 0] > lower_bound[0]),
                    (imageHSV[:, :, 0] < upper_bound[0]),
                ),
                np.logical_and(
                    (imageHSV[:, :, 1] > lower_bound[1]),
                    (imageHSV[:, :, 1] < upper_bound[1]),
                ),
            ),
            np.logical_and(
                (imageHSV[:, :, 2] > lower_bound[2]),
                (imageHSV[:, :, 2] < upper_bound[2]),
            ),
        )
    )

    color_value = nb_pixels_in_range / nb_pixels_total
    return color_value


def get_value_from_rgb(imageRGB, color_index):
    # pixel is good if value from color index is greater then other
    nb_good_pixels = 0
    nb_total_pixels = 0
    for i in range(imageRGB.shape[0]):
        for j in range(imageRGB.shape[1]):
            nb_total_pixels += 1
            good_value = imageRGB[i, j, color_index]
            if good_value >= np.max(imageRGB[i, j]):
                nb_good_pixels += 1

    return nb_good_pixels / nb_total_pixels


def get_value_from_rgb_shelby(imageRGB, color_index):

    pixel_most = np.zeros(3)
    nb_pixels = np.shape(imageRGB)[0] * np.shape(imageRGB)[1]
    buffer = 10
    pixel_most[0] = np.count_nonzero(
        np.logical_and(
            (imageRGB[:, :, 0] > imageRGB[:, :, 2] + buffer),
            (imageRGB[:, :, 0] > imageRGB[:, :, 1] + buffer),
        )
    )
    pixel_most[1] = np.count_nonzero(
        np.logical_and(
            (imageRGB[:, :, 1] > imageRGB[:, :, 0] + buffer),
            (imageRGB[:, :, 1] > imageRGB[:, :, 2] + buffer),
        )
    )
    pixel_most[2] = np.count_nonzero(
        np.logical_and(
            (imageRGB[:, :, 2] > imageRGB[:, :, 0] + buffer),
            (imageRGB[:, :, 2] > imageRGB[:, :, 1] + buffer),
        )
    )
    return pixel_most[color_index] / nb_pixels


def get_square_value(imageRGB):
    # convert to gray
    imageGray = skic.rgb2gray(imageRGB)
    corner = corner_peaks(corner_fast(imageGray), min_distance=10)
    # print(corner)
    return len(corner)


def get_light_pixel_top_image(imageRGB):
    value = np.count_nonzero(
        np.logical_and(
            np.logical_and(
                (imageRGB[0:45, :, 0] >= 200), (imageRGB[0:45, :, 1] >= 200)
            ),
            (imageRGB[0:45, :, 2] >= 200),
        )
    )
    return value / (45 * imageRGB.shape[1])
