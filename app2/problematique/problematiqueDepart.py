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
    N = 6
    im_list = np.sort(random.sample(range(np.size(ImageCollection.image_list, 0)), N))
    print(im_list)
    ImageCollection.images_display(im_list)
    ImageCollection.view_histogrammes(im_list)
    plt.show()


######################################
if __name__ == '__main__':
    main()
