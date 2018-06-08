import numpy                as np
import pandas               as pd
from sklearn.neighbors     import KNeighborsClassifier

import dirs
import defines              as defs

def nearest_neighbours(x_train, y_train, x_test, y_test):
    '''
        Train K-Nearest Neighbours classifier on x_train and predict on x_test.

        x_train, x_test: DataFrames of shape data x features.
    '''
    # TODO: Experiment with 'weights' parameter
    # classWeights = {defs.posCode: 0.5, defs.negCode: 0.5}
    knn = KNeighborsClassifier( n_neighbors=5, algorithm='ball_tree', weights='uniform',
                                p=2, metric='minkowski', n_jobs=-1)

    # print("\nParameters initialization:")
    # print(percep.coef_)

    knn.fit(x_train, y_train)#, weights) # TODO: Add class weights

    predictions = knn.predict(x_test)

    return predictions

