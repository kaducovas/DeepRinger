import numpy                as np
import pandas               as pd
from sklearn.tree           import DecisionTreeClassifier
from sklearn.ensemble       import RandomForestClassifier, AdaBoostClassifier

import dirs
import defines              as defs

def decision_tree(x_train, y_train, x_test, y_test):
    '''
        Train a Decision Tree classifier on x_train and predict on x_test.

        x_train, x_test: DataFrames of shape num_samples x num_features.
    '''

    classWeights = {defs.posCode: 0.5, defs.negCode: 0.5}
    tree = DecisionTreeClassifier(class_weight=classWeights, criterion='entropy', max_depth=15, min_samples_leaf=5)

    tree.fit(x_train, y_train)

    predictions = tree.predict(x_test)

    return predictions

def random_forest(x_train, y_train, x_test, y_test):
    '''
        Train an ensemble of Decision Trees on x_train and predict on x_test.

        x_train, x_test: DataFrames of shape num_samples x num_features.
    '''

    classWeights = {defs.posCode: 0.5, defs.negCode: 0.5}
    forest = RandomForestClassifier(n_estimators=50, class_weight=classWeights, criterion='gini', max_depth=15, min_samples_leaf=5, n_jobs=-1)

    forest.fit(x_train, y_train)

    # print(np.sort(forest.feature_importances_))

    predictions = forest.predict(x_test)

    return predictions

def ada_boost(x_train, y_train, x_test, y_test):
    '''
        Train an AdaBoost ensemble of Decision Trees on x_train and predict on x_test.

        x_train, x_test: DataFrames of shape num_samples x num_features.
    '''

    classWeights = {defs.posCode: 0.5, defs.negCode: 0.5}
    ada = AdaBoostClassifier(n_estimators=50, learning_rate=1.0)

    ada.fit(x_train, y_train)

    # print(np.sort(forest.feature_importances_))

    predictions = ada.predict(x_test)

    return predictions

