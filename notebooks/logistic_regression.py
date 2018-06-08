import numpy                as np
import pandas               as pd
from sklearn.linear_model   import LogisticRegression

import dirs
import defines              as defs

def log_reg(x_train, y_train, x_test, y_test):
    '''
        Train Logistic Regression classifier on x_train and predict on x_test.

        x_train, x_test: DataFrames of shape data x features.
        thres: Class discrimination threshold
    '''

    logReg = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', n_jobs=1, max_iter=100)

    logReg.fit(x_train, y_train)#, weights) # TODO: Add class weights

    predictions = logReg.predict(x_test)

    return predictions

def ridge_log_reg(x_train, y_train, x_test, y_test, reg=1.0):
    '''
        Train Logistic Regression classifier on x_train and predict on x_test.

        x_train, x_test: DataFrames of shape data x features.
        reg: Regularization parameter. Multiplies penalty by 1/reg, so smaller
        values mean stronger regularization.
    '''

    logReg = LogisticRegression(penalty='l2', C=reg, solver='liblinear', n_jobs=1, max_iter=100)

    logReg.fit(x_train, y_train)#, weights) # TODO: Add class weights

    predictions = logReg.predict(x_test)

    return predictions
