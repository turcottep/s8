"""
Authors: Antoine Marion 2021, JBM major refactor 2021 & 2022

Fonctions de classification pour le labo
S8 GIA APP2
TODO modifier si nécessaire pour la problématique

Fichier comporte 3 sections principales
A) Classificateurs de base: fonctions au prototype relativement similaire, prennent des données d'entraînement
        pour créer un modèle, et ensuite calculent la prédiction à partir de ce modèle pour 1 dataset + 1 autre optionnel
    prototype: options d'algo, données d'entraînement + étiquette, données de test 1, données de test 2 (opt)
    i. compute_prob_dens_gaussian: calcule un modèle gaussien pour chaque classe des données d'entraînement et calcule
        la probablité pour chaque donnée des sets de test, pour chaque modèle de classe
    ii. ppv_classify: utilise les données d'entraînement comme représentants de classes et exécute un k-ppv sur les sets
        de test
    iibis. kmean_alg: produit des représentants de classes (pas une classification)
    iii. nn_classify: entraîne un RN à partir des données d'entraînements et prédit les sets de test

B) Wrappers pour les classificateurs de base: encore une fois, fonctions au prototype relativement similaires,
        enveloppent les classificateurs ci-dessus avec du nice-to-have, en particulier du formattage des prédictions,
        calcul de taux d'erreur et affichage de graphiques de résultats
    prototype général: options d'algo, données d'entraînement + étiquettes, données de test aléatoires pour visualiser
        les frontières, option de graphique, données de test 2 + étiquettes (opt, pour quantifier la performance)
    i. full_Bayes_risk: assume des coûts unitaires et des apriori égaux
    ii. full_ppv
    iibis. full_kmean
    iii. full_nn

C) Autres fonctions modulaires
    i. helpers génériques
        - calc_erreur_classification: compare 2 vecteurs et retourne les index des éléments différents
    ii. helpers pour modèles gaussiens
        - get_borders: permet de calculer l'équation de la frontière entre chaque paire de classes d'entrée,
                en assumant un modèle gaussien; voir l'exercice préparatoire du labo 2
    iii. helpers pour les RN
        - print_every_N_epochs: callback custom pour un affichage plus convivial pendant l'entraînement
"""

from itertools import combinations
import numpy as np

from sklearn.cluster import KMeans as km
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.model_selection import train_test_split as ttsplit
from sklearn.preprocessing import OneHotEncoder

import keras as K
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam

import helpers.analysis as an


def compute_prob_dens_gaussian(train_data, test_data1, test_data2):
    """
    Construit les modèles gaussien de chaque classe (première dimension de train_data)
    puis calcule la densité de probabilité de chaque point dans test_data par rapport à chaque classe

    retourne un tableau de la valeur de la densité de prob pour chaque point dans test_data1 et un autre pour
        test_data2 par rapport à chaque classe
    """
    train_data = np.array(train_data)
    x, y, z = train_data.shape

    # bâtit la liste de toutes les stats
    # i.e. un modèle
    # donc ceci correspond à .fit dans la logique sklearn
    mean_list = []
    cov_list = []
    det_list = []
    inv_cov_list = []
    for i in range(x):
        mean, cov, pouet, pouet  = an.calcModeleGaussien(train_data[i])
        mean_list.append(mean)
        inv_cov = np.linalg.inv(cov)
        cov_list.append(cov)
        inv_cov_list.append(inv_cov)
        det = np.linalg.det(cov)
        det_list.append(det)

    # calcule les probabilités de chaque point des données de test pour chaque classe
    # correspond à .predict dans la logique sklearn
    test_data1 = np.array(test_data1)
    t1, v1 = test_data1.shape
    test_data2 = np.array(test_data2)
    t2, v2 = test_data2.shape
    dens_prob1 = []
    dens_prob2 = []
    # calcule la valeur de la densité de probabilité pour chaque point de test
    for i in range(x):  # itère sur toutes les classes
        # pour les points dans test_data1
        # TODO L2.E2.3 Compléter le calcul ici
        mahalanobis1 = np.array([1 for j in range(t1)])
        prob1 = 1 / np.sqrt(det_list[i] * (2 * np.pi) ** z) * np.exp(-mahalanobis1 / 2)
        dens_prob1.append(prob1)
        # pour les points dans test_data2
        mahalanobis2 = np.array([1 for j in range(t2)])
        prob2 = 1 / np.sqrt(det_list[i] * (2 * np.pi) ** z) * np.exp(-mahalanobis2 / 2)
        dens_prob2.append(prob2)

    return np.array(dens_prob1).T, np.array(dens_prob2).T  # reshape pour que les lignes soient les calculs pour 1 point original


def ppv_classify(n_neighbors, train_data, classes, test1, test2=None):
    """
    Classifie test1 et test2 dans les classes contenues dans train_data et étiquetées dans "classes"
        au moyen des n_neighbors plus proches voisins (distance euclidienne)
    Retourne les prédictions pour chaque point dans test1, test2
    Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    """
    # Creation classificateur
    # n_neighbors est le nombre k
    # metric est le type de distance entre les points. La liste est disponible ici:
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html#sklearn.neighbors.DistanceMetric
    # TODO L2.E3.1 Compléter la logique pour utiliser la librairie ici
    kNN = knn(1, metric='minkowski')  # minkowski correspond à distance euclidienne lorsque le paramètre p = 2
    predictions_test1 = np.zeros(len(test1))  # classifie les données de test1
    predictions_test2 = np.zeros(len(test2)) if np.asarray(test2).any() else np.asarray([])  # classifie les données de test2 si présentes
    return predictions_test1, predictions_test2


def kmean_alg(n_clusters, data):
    """
    Calcule n_clusters représentants de classe pour les classes contenues dans data (première dimension)
    Retourne la suite des représentants de classes et leur étiquette
    Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    """
    data = np.array(data)
    x, y, z = data.shape

    cluster_centers = []
    cluster_labels = np.zeros((n_clusters * x, 1))
    # calcule les représentants pour chaque classe séparément
    for i in range(x):
        # TODO L2.E3.3 compléter la logique pour utiliser la librairie ici
        kmeans_C = km(1)
        kmeans_C.fit(np.array(data[i]))
        cluster_centers.append(kmeans_C.cluster_centers_)
        cluster_labels[range(n_clusters * i, n_clusters * (i + 1))] = i  # assigne la classe en ordre ordinal croissant

    if n_clusters == 1:  # gère les désagréments Python
        cluster_centers = np.array(cluster_centers)[:, 0]
    else:
        cluster_centers = np.array(cluster_centers)
        x, y, z = cluster_centers.shape
        cluster_centers = cluster_centers.reshape(x * y, z)
    return cluster_centers, cluster_labels


def nn_classify(n_hidden_layers, n_neurons, train_data, classes, test1, test2=None):
    """
    Classifie test1 et test2 au moyen d'un réseau de neurones entraîné avec train_data et les sorties voulues "classes"
    Retourne les prédictions pour chaque point dans test1, test2
    """
    # (e.g. filtering, normalization, dimensionality reduction)
    data, minmax = an.scaleData(train_data)

    # Convertit la représentation des étiquettes pour utiliser plus facilement la cross-entropy comme loss
    # TODO L3.E2.1
    encoder = OneHotEncoder(sparse=False)
    targets = classes

    # Crée des ensembles d'entraînement et de validation
    # TODO L3.E2.3
    training_data = data
    training_target = targets
    validation_data = []
    validation_target = []

    # Create neural network
    # TODO L3.E2.6 Tune the number and size of hidden layers
    NNmodel = Sequential()
    NNmodel.add(Dense(units=n_neurons, activation='tanh', input_shape=(data.shape[-1],)))
    for i in range(2, n_hidden_layers):
        NNmodel.add(Dense(units=n_neurons, activation='tanh'))
    NNmodel.add(Dense(units=targets.shape[-1], activation='tanh'))
    print(NNmodel.summary())

    # Define training parameters
    # TODO L3.E2.6 Tune the training parameters
    # TODO L3.E2.1
    NNmodel.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # Perform training
    # TODO L3.E2.4
    callback_list = [K.callbacks.EarlyStopping(patience=50, verbose=1, restore_best_weights=1), print_every_N_epochs(25)]
    # TODO L3.E2.6 Tune the maximum number of iterations and desired error
    # TODO L3.E2.2 L3.E2.3
    NNmodel.fit(training_data, training_target, batch_size=len(data), verbose=1,
              epochs=10, shuffle=True, callbacks=[])

    # Save trained model to disk
    NNmodel.save('3classes.h5')
    an.plot_metrics(NNmodel)

    # Test model (loading from disk)
    # TODO problématique: implement a mechanism to keep the best model and/or compare model performance across training runs
    NNmodel = load_model('3classes.h5')

    # classifie les données de test
    # decode la sortie one hot en numéro de classe 0 à N directement
    predictions_test1 = np.argmax(NNmodel.predict(an.scaleDataKnownMinMax(test1, minmax)), axis=1)
    predictions_test2 = np.argmax(NNmodel.predict(an.scaleDataKnownMinMax(test2, minmax)), axis=1) \
        if np.asarray(test2).any() else []  # classifie les données de test2 si présentes
    return predictions_test1, predictions_test2


def full_Bayes_risk(train_data, train_classes, donnee_test, title, extent, test_data, test_classes):
    """
    Classificateur de Bayes complet pour des classes équiprobables (apriori égal)
    Selon le calcul direct du risque avec un modèle gaussien
    Produit un graphique pertinent et calcule le taux d'erreur moyen

    train_data: données qui servent à bâtir les modèles
    train_classes: étiquettes de train_data
    test_data: données étiquetées dans "classes" à classer pour calculer le taux d'erreur
    test_classes: étiquettes de test_data
    donnee_test: données aléatoires pour visualiser la frontière
    title: titre à utiliser pour la figure
    """

    # calcule p(x|Ci) pour toutes les données étiquetées
    # rappel (c.f. exercice préparatoire)
    # ici le risque pour la classe i est pris comme 1 - p(x|Ci) au lien de la somme du risque des autres classes
    prob_dens, prob_dens2 = compute_prob_dens_gaussian(train_data, donnee_test, test_data)
    # donc minimiser le risque revient à maximiser p(x|Ci)
    classified = np.argmax(prob_dens, axis=1).reshape(len(donnee_test), 1)
    classified2 = np.argmax(prob_dens2, axis=1).reshape(test_classes.shape)

    # calcule le taux de classification moyen
    error_class = 6  # optionnel, assignation d'une classe différente à toutes les données en erreur, aide pour la visualisation
    error_indexes = calc_erreur_classification(test_classes, classified2)
    classified2[error_indexes] = error_class
    print(
        f'Taux de classification moyen sur l\'ensemble des classes, {title}: {100 * (1 - len(error_indexes) / len(classified2))}%')

    train_data = np.array(train_data)
    x, y, z = train_data.shape
    train_data = train_data.reshape(x*y, z)
    #  view_classification_results(train_data, test1, c1, c2, glob_title, title1, title2, extent, test2=None, c3=None, title3=None)
    an.view_classification_results(train_data, donnee_test, train_classes, classified / error_class / .75,
                                   f'Classification de Bayes, {title}', 'Données originales', 'Données aléatoires',
                                   extent, test_data, classified2 / error_class / .75, 'Données d\'origine reclassées')


def full_ppv(n_neighbors, train_data, train_classes, datatest1, title, extent, datatest2=None, classestest2=None):
    """
    Classificateur PPV complet
    Utilise les données de train_data étiquetées dans train_classes pour créer un classificateur n_neighbors-PPV
    Trie les données de test1 (non étiquetées), datatest2 (optionnel, étiquetées dans "classestest2"
    Calcule le taux d'erreur moyen pour test2 le cas échéant
    Produit un graphique des résultats pour test1 et test2 le cas échéant
    """
    predictions, predictions2 = ppv_classify(n_neighbors, train_data, train_classes.ravel(), datatest1, datatest2)
    predictions = predictions.reshape(len(datatest1), 1)

    error_class = 6  # optionnel, assignation d'une classe différente à toutes les données en erreur, aide pour la visualisation
    if np.asarray(datatest2).any():
        predictions2 = predictions2.reshape(len(datatest2), 1)
        # calcul des points en erreur à l'échelle du système

        error_indexes = calc_erreur_classification(classestest2, predictions2.reshape(classestest2.shape))
        predictions2[error_indexes] = error_class
        print(
            f'Taux de classification moyen sur l\'ensemble des classes, {title}: {100 * (1 - len(error_indexes) / len(classestest2))}%')
    #  view_classification_results(train_data, test1, c1, c2, glob_title, title1, title2, extent, test2=None, c3=None, title3=None)
    an.view_classification_results(train_data, datatest1, train_classes, predictions, title, 'Représentants de classe',
                                   f'Données aléatoires classées {n_neighbors}-PPV',
                                   extent, datatest2, predictions2 / error_class / 0.75,
                                   f'Prédiction de {n_neighbors}-PPV, données originales')


def full_kmean(n_clusters, train_data, train_classes, title, extent):
    """
    Exécute l'algorithme des n_clusters-moyennes sur les données de train_data étiquetées dans train_classes
    Produit un graphique des représentants de classes résultants
    Retourne les représentants de classe obtenus et leur étiquette respective
    """
    cluster_centers, cluster_labels = kmean_alg(n_clusters, train_data)

    train_data = np.array(train_data)
    x, y, z = train_data.shape
    train_data = train_data.reshape(x * y, z)

    #  view_classification_results(train_data, test1, c1, c2, glob_title, title1, title2, extent, test2=None, c3=None, title3=None)
    an.view_classification_results(train_data, cluster_centers, train_classes, cluster_labels, title, 'Données d\'origine',
                                   f'Clustering de {n_clusters}-Means', extent)

    return cluster_centers, cluster_labels


def full_nn(n_hiddenlayers, n_neurons, train_data, train_classes, test1, title, extent, test2=None, classes2=None):
    """
    Classificateur RNA complet
    Utilise les données de train_data étiquetées dans train_classes pour entraîner un réseau de neurones
    Trie les données de test1 (non étiquetées), datatest2 (optionnel, étiquetées dans "classestest2"
    Calcule le taux d'erreur moyen pour test2 le cas échéant
    Produit un graphique des résultats pour test1 et test2 le cas échéant
    """
    predictions, predictions2 = nn_classify(n_hiddenlayers, n_neurons, train_data, train_classes.ravel(), test1, test2)
    predictions = predictions.reshape(len(test1), 1)

    error_class = 6  # optionnel, assignation d'une classe différente à toutes les données en erreur, aide pour la visualisation
    if np.asarray(test2).any():
        predictions2 = predictions2.reshape(len(test2), 1)
        # calcul des points en erreur à l'échelle du système
        error_indexes = calc_erreur_classification(classes2, predictions2)
        predictions2[error_indexes] = error_class
        print(f'Taux de classification moyen sur l\'ensemble des classes, {title}: {100 * (1 - len(error_indexes) / len(classes2))}%')
    #  view_classification_results(train_data, test1, c1, c2, glob_title, title1, title2, extent, test2=None, c3=None, title3=None)
    an.view_classification_results(train_data, test1, train_classes, predictions, title, 'Données originales',
                                   f'Données aléatoires classées par le RNA',
                                   extent, test2, predictions2 / error_class / 0.75,
                                   f'Prédiction du RNA, données originales')


def calc_erreur_classification(original_data, classified_data):
    """
    Retourne le nombre d'éléments différents entre deux vecteurs
    """
    # génère le vecteur d'erreurs de classification
    vect_err = np.absolute(original_data - classified_data).astype(bool)
    indexes = np.array(np.where(vect_err == True))[0]
    print(f'\n\n{len(indexes)} erreurs de classification sur {len(original_data)} données')
    # print(indexes)
    return indexes


def get_borders(data):
    """
    ***Pas validé sur des classes autres que les classes du laboratoire
    Calcule les frontières numériques entre n classes de dimension 2 en assumant un modèle gaussien

    data format: [C1, C2, C3, ... Cn]
    retourne 1 liste:
        border_coeffs: coefficients numériques des termes de l'équation de frontières
            [x**2, xy, y**2, x, y, cst (cote droit de l'equation de risque), cst (dans les distances de mahalanobis)]

    Le calcul repose sur une préparation analytique qui conduit à
    g(y) = y*A*y + b*y + C          avec
    y la matrice des dimensions d'1 vecteur de la classe
    et pour chaque paire de classe C1 C2:
    A = inv(cov_1) - inv(cov_2)
    b = -2*(inv(cov_2)*m2 - inv(cov_1)*m1)
    C = c+d
    d = -(transp(m1)*inv(cov_1)*m1 - transp(m2)*inv(cov_2)*m2)
    c = -ln(det(cov_2)/det(cov_1))
    """
    # Initialisation des listes
    # Portion numérique
    avg_list = []
    cov_list = []
    det_list = []
    inv_cov_list = []
    border_coeffs = []

    # calcul des stats des classes
    for i in range(len(data)):
        # stats de base
        avg, cov, pouet, pouet = an.calcModeleGaussien(data[i])
        avg_list.append(avg)
        inv_cov = np.linalg.inv(cov)
        cov_list.append(cov)
        inv_cov_list.append(inv_cov)
        det = np.linalg.det(cov)
        det_list.append(det)

    # calcul des frontières
    for item in combinations(range(len(data)), 2):
        # les coefficients sont tirés de la partie préparatoire du labo
        # i.e. de la résolution analytique du risque de Bayes
        # partie numérique
        a = np.array(inv_cov_list[item[1]] - inv_cov_list[item[0]])
        b = -np.array([2 * (np.dot(inv_cov_list[item[1]], avg_list[item[1]]) - np.dot(inv_cov_list[item[0]], avg_list[item[0]]))])
        d = -(np.dot(np.dot(avg_list[item[0]], inv_cov_list[item[0]]), np.transpose(avg_list[item[0]])) -
              np.dot(np.dot(avg_list[item[1]], inv_cov_list[item[1]]), np.transpose(avg_list[item[1]])))
        c = -np.log(det_list[item[1]] / det_list[item[0]])

        # rappel: coef order: [x**2, xy, y**2, x, y, cst (cote droit log de l'equation de risque), cst (dans les distances de mahalanobis)]
        border_coeffs.append([a[0, 0], a[0, 1] + a[1, 0], a[1, 1], b[0, 0], b[0, 1], c, d])
        # print(border_coeffs[-1])

    return border_coeffs


class print_every_N_epochs(K.callbacks.Callback):
    """
    Helper callback pour remplacer l'affichage lors de l'entraînement
    """
    def __init__(self, N_epochs):
        self.epochs = N_epochs

    def on_epoch_end(self, epoch, logs=None):
        # TODO L3.E2.4
        if True:
            print("Epoch: {:>3} | Loss: ".format(epoch) +
                  f"{logs['loss']:.4e}" + " | Valid loss: " + f"{logs['val_loss']:.4e}" +
                  (f" | Accuracy: {logs['accuracy']:.4e}" + " | Valid accuracy " + f"{logs['val_accuracy']:.4e}"
                   if 'accuracy' in logs else "") )
