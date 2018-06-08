import numpy                as np
import pandas               as pd
from sklearn.linear_model   import Perceptron

import dirs
import defines              as defs

def perceptron(x_train, y_train, x_test, y_test):
    '''
        Train a Perceptron classifier with linear activation function on x_train
        and predict on x_test.

        x_train, x_test: DataFrames of shape data x features.
    '''

    classWeights = {defs.posCode: 0.5, defs.negCode: 0.5}
    percep = Perceptron(shuffle=True, n_jobs=-1, class_weight=classWeights, max_iter=1000, tol=1e-3)

    # print("\nParameters initialization:")
    # print(percep.coef_)

    percep.fit(x_train, y_train)#, weights) # TODO: Add class weights

    predictions = percep.predict(x_test)

    return predictions

