import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import keras
import sklearn
import vispy
import tensorflow

def corr_matrix_plot(inputDf):
    '''
    inputDf is an observations by features DataFrame
    '''
    corr = inputDf.corr()

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(13, 11))

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, cmap='viridis', vmax=.3, center=0,
                square=True, linewidths=0.01, cbar_kws={"shrink": .5})

    ax.set_title("Correlation plot of {} features".format(len(corr)))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()
    return fig, ax

def dimension_reduction(dataDf, keepComp=0):
    '''
    dataDf:   is an observations by features DataFrame
    keepComp: Number of components to keep
    '''
    if keepComp <= 0:
        keepComp = dataDf.shape[1]

    dataPCA = PCA(n_components=None)

    xCompact = dataPCA.fit_transform(dataDf)

    explVar = dataPCA.explained_variance_ratio_
    # print("\nExplained variance:\n", explVar)
    print("\nN components:", dataPCA.n_components_)
    print("\nPrincipal components to keep: ", keepComp)

    # # Save compact data as the reference DataFrame
    # yCol = dataDf.iloc[:, -1].values.reshape((dataDf.iloc[:, -1].shape[0], 1))
    # compactDf = pd.DataFrame(np.concatenate((xCompact[:, :keepComp], yCol), axis=1))
    compactDf = pd.DataFrame(xCompact[:, :keepComp])
    print("\nCompact data: ", np.shape(compactDf))
    return compactDf

def eigen_plot(inputDf, labels):
    # Get Principal Components ratio
    eignVals =  np.linalg.svd(inputDf, compute_uv=False)
    eignVals = eignVals/eignVals.sum()

    features = len(eignVals)
    index    = list(range(features))

    # Compute cumulative contributions
    sumVals = np.zeros(features)
    for i in range(features):
        sumVals[i] = np.sum(eignVals[:i])

    sumVals = sumVals[1:]
    sumVals = np.append(sumVals, 1.0)

    # Plot barplot of Principal Components ratios
    ax = sns.barplot(x=index, y=eignVals, palette="Blues_d")

    # Plot lineplot of cumulative contributions
    ax.plot(index, sumVals, 'b-')

    ax.set_title("{} Principal values".format(features))
    ax.set_ylim(bottom=0.0, top=1.0)
    ax.get_xaxis().set_visible(False)

    plt.show()
    return ax

def load_dataset(path, fracPos=1.0, fracNeg=1.0):
    # Load raw data
    data = np.load(path)

    # Save each class
    # classPos = data[data.keys()[0]]
    # classNeg = data[data.keys()[1]]
    classPos = pd.DataFrame(data[data.keys()[0]])
    classNeg = pd.DataFrame(data[data.keys()[1]])

    # Sample a fraction of the data
    # doing this before anything else avoids memory issues
    features = classPos.shape[1]
    totPos = classPos.shape[0]
    totNeg = classNeg.shape[0]

    # choose random indexes
    # indexPos = np.random.choice(range(totPos), size=int(totPos*fracPos), replace=False)
    # indexNeg = np.random.choice(range(totNeg), size=int(totNeg*fracNeg), replace=False)
    #
    # classPos = classPos[indexPos,:]
    # classNeg = classNeg[indexNeg,:]

    classPos = classPos.sample(frac=fracPos, axis=0)
    classNeg = classNeg.sample(frac=fracNeg, axis=0)

    entriesPos = classPos.shape[0]
    entriesNeg = classNeg.shape[0]
    total = entriesPos + entriesNeg

    # Create Label vectors
    yPos  =  np.ones((entriesPos,1))  # [+1, -1] representation has zero mean, which
    yNeg  = -np.ones((entriesNeg,1))  # can lead to faster gradient convergence

    # Concatenate input data and labels
    labels = np.concatenate((yPos, yNeg))
    # classPos = pd.DataFrame(np.concatenate((classPos, yPos), axis=1))
    # classNeg = pd.DataFrame(np.concatenate((classNeg, yNeg), axis=1))

    print("\nData loaded with following class distribution: ")
    print("Positive class: {:.2f} %, {} entries ".format(entriesPos/total*100, entriesPos))
    print("Negative class: {:.2f} %, {} entries ".format(entriesNeg/total*100, entriesNeg))

    # Save dataset in a DataFrame
    dataDf = pd.DataFrame(np.concatenate((classPos, classNeg)))
    dataDf.rename({-1:"labels"}, axis='columns')
    return dataDf, labels

def plot_3d(compactDf, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    posData = compactDf[labels == +1]
    negData = compactDf[labels == -1]

    ax.scatter(posData.iloc[:, 0], posData.iloc[:, 1], posData.iloc[:, 2], c='xkcd:twilight blue', alpha=0.3)
    ax.scatter(negData.iloc[:, 0], negData.iloc[:, 1], negData.iloc[:, 2], c='xkcd:fire engine red', alpha=0.3)

    ax.set_title("3D Scatter Plot")
    plt.show()
    return fig, ax

def plot_boxplot(inputDf, labels):
    # posClass = dataDf.loc[dataDf.iloc[:, -1] == +1]
    # negClass = dataDf.loc[dataDf.iloc[:, -1] == -1]

    # ax = sns.boxplot(x='Variables', y='Values', data=dataDf.loc[:, -1].values)
    dataDf = pd.concat([inputDf, pd.DataFrame(labels, columns=['labels'])])

    ax = sns.boxplot(data=inputDf)
    ax.set_title("Boxplot of {} features".format(inputDf.shape[1]))
    plt.show()

    # Unusable, scales very badly with large number of observations
    # ax = sns.swarmplot(data=inputDf)
    # ax.set_title("Swarmplot of {} features".format(inputDf.shape[1]))
    plt.show()
    return ax

def preproc(dataDf, verbose=False):
    ## Display basic dataset information, clean up and preprocess the data

    if verbose:
        print("\nData shape: ", dataDf.shape)

    # Number of zero values per feature
    zeroValues = (dataDf == 0).sum(axis=0)
    allZeros   = (dataDf == 0).all(axis=0).sum()
    # print("\nNumber of zero-valued entries per feature:\n", zeroValues)

    # Features containing only zeros will be dropped
    dataDf = dataDf.loc[:, (dataDf != 0).any(axis=0)]
    dataDf = dataDf.reset_index(drop=True)
    print("\n{} features containing only zeros have been dropped from data.".format(allZeros))
    if verbose:
        print("\nNew data shape: ", dataDf.shape)

    # Z-score normalization
    mean = dataDf.mean(axis=0)
    std  = dataDf.std (axis=0)
    dataDf = (dataDf - mean)/std

    ## "Basic Sample Statistics"
    if verbose:
        print("\nMax:\n", dataDf.max   (axis=0))
        print("\nMin:\n", dataDf.min   (axis=0))
        print("\nMean:\n",dataDf.mean  (axis=0))
        print("\nMed:\n", dataDf.median(axis=0))
        print("\nVar:\n", dataDf.std   (axis=0))
        print("\nStd:\n", dataDf.var   (axis=0))

    return dataDf

def projection_plot(inputDf, labels):
    '''
    inputDf is an observations by features DataFrame
    labels is an observations by 1 DataFrame of [+1, -1] labels
    '''
    features = inputDf.shape[1]
    # features = 10

    posData = inputDf[labels == +1]  # Last DataFrame columns are labels:
    negData = inputDf[labels == -1]  # used to filter data by class

    fig, axs = plt.subplots(nrows=features, ncols=features, figsize=(12,10))#, squeeze=False, tight_layout=True)

    # print("\nMax: ", features)
    for row in range(features):
        for col in range(features):
            if row <= col:
                if row == col:
                    # Plot histogram of feature i
                    axs[row,col].hist(inputDf.iloc[:, col].values, bins='auto', color='xkcd:dull blue')  # axis.hist() method doesn't work with DataFrame
                else:
                    # Plot projection X_i by X_j
                    axs[row,col].plot(posData.iloc[:, row], posData.iloc[:, col], '.', alpha=0.3, markerfacecolor='xkcd:fire engine red', markeredgecolor='xkcd:tomato')
                    axs[row,col].plot(negData.iloc[:, row], negData.iloc[:, col], '.', alpha=0.3, markerfacecolor='xkcd:night blue', markeredgecolor='xkcd:dark blue')

            # Hide axis labels
            axs[row,col].get_xaxis().set_visible(False)
            axs[row,col].get_yaxis().set_visible(False)
            axs[row,col].set_yticklabels([])
            axs[row,col].set_xticklabels([])

            # Set up border labels
            if col == 0:
                axs[row,col].set_ylabel("X{}".format(row))
                axs[row,col].get_yaxis().set_visible(True)
            if row == (features-1):
                axs[row,col].set_xlabel("X{}".format(col))
                axs[row,col].get_xaxis().set_visible(True)

    plt.show()
    return axs, fig
