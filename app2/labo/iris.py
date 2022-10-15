# Copyright (c) 2018, Simon Brodeur
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  - Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  - Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Author: Simon Brodeur <simon.brodeur@usherbrooke.ca>
# Université de Sherbrooke, APP3 S8GIA, A2018

"""
Standalone example
Classificateur de fleurs basé sur des caractéristiques mesurées (représentation mesurée)
S8 GIA APP2
TODO voir L3.E3
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

import keras as K
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split as ttsplit

import helpers.analysis as an
import helpers.classifiers as classifiers


def main():
    # Load iris data set from file
    # Attributes are: petal length, petal width, sepal length, sepal width
    # TODO: Analyze the input data
    S = scipy.io.loadmat('iris.mat')
    data = np.array(S['data'], dtype=np.float32)
    target = np.array(S['target'], dtype=np.float32)

    target_decode = np.argmax(target, axis=-1) # targets are 1hot encoded
    # sépare les classes pour en afficher les propriétés
    C1 = data[np.where(target_decode == 0)]
    C2 = data[np.where(target_decode == 1)]
    C3 = data[np.where(target_decode == 2)]
    an.calcModeleGaussien(C1, '\nClasse versicolor')
    an.calcModeleGaussien(C2, '\nClasse virginica')
    an.calcModeleGaussien(C3, '\nClasse setose')

    # Show the 3D projection of the data
    # TODO L3.E3.1 Observez si différentes combinaisons de dimensions sont discriminantes
    data3D = data[:, 1:4]
    an.view3D(data3D, target_decode, 'dims 1 2 3')
    data3D = data[:, [0,2,3]]
    an.view3D(data3D, target_decode, 'dims 0 2 3')

    # TODO : Apply any relevant transformation to the data
    # TODO L3.E3.1 Conservez les dimensions qui vous semblent appropriées et décorrélées-les au besoin
    # (e.g. filtering, normalization, dimensionality reduction)
    data, minmax = an.scaleData(data)

    # TODO L3.E3.4
    training_data = data
    training_target = target
    validation_data = []
    validation_target = []

    # Create neural network
    # TODO L3.E3.3  Tune the number and size of hidden layers
    model = Sequential()
    model.add(Dense(units=50, activation='tanh',
                    input_shape=(data.shape[-1],)))
    model.add(Dense(units=target.shape[-1], activation='linear'))
    print(model.summary())

    # Define training parameters
    # TODO L3.E3.3 Tune the training parameters
    model.compile(optimizer=SGD(learning_rate=0.1, momentum=0.9), loss='mse')

    # Perform training
    callback_list = []
    # TODO L3.E3.3  Tune the training hyperparameters
    model.fit(training_data, training_target, batch_size=len(data), verbose=1,
              epochs=1000, shuffle=True, callbacks=callback_list)

    # Save trained model to disk
    model.save('iris.h5')

    an.plot_metrics(model)

    # Test model (loading from disk)
    model = load_model('iris.h5')
    targetPred = model.predict(data)

    # Print the number of classification errors from the training data
    error_indexes = classifiers.calc_erreur_classification(np.argmax(targetPred, axis=-1), target_decode)

    plt.show()

if __name__ == "__main__":
    main()
