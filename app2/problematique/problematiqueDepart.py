"""
Script de départ de la problématique
Problématique APP2 Module IA S8
"""

import matplotlib.pyplot as plt
import numpy as np
import random


from ImageCollection import ImageCollection


#######################################
def main():
    # Génère une liste de N images, les visualise et affiche leur histo de couleur
    # TODO: voir L1.E3 et problématique
    image_collection = ImageCollection()
    # N = 6
    # im_list = np.sort(random.sample(range(np.size(ImageCollection.image_list, 0)), N))
    # print(im_list)
    # ImageCollection.images_display(im_list)
    # ImageCollection.view_histogrammes(im_list)

    # ImageCollection.get_all_noise_levels(im_list)
    # ImageCollection.get_color_values(im_list, 255, 0, 0)

    image_collection.check_discrimination()

    # image_collection.test_square()

    # image_collection.classify_images_bayes()
    # image_collection.classify_images_knn()
    # image_collection.classify_images_nn()

    plt.show()


######################################
if __name__ == "__main__":
    main()
