import numpy                as np
import pandas               as pd
from sklearn.naive_bayes    import GaussianNB

import dirs

def gaussian_naive_bayes(dataDf, labels, testDf, testLabels):
    '''
        Train Naive Bayes classifier on dataDf and predict on testDf
        dataDf, testDf: DataFrames of shape (data x features)
    '''
    gnb = GaussianNB()
    gnb.fit(dataDf, labels)

    predictions = gnb.predict(testDf)

    return predictions

