import numpy                as np
import pandas               as pd

import dirs
from load_dataset           import load_dataset
from preproc                import preproc, dimension_reduction
from vis_functions          import (eigen_plot,
                                    plot_3d,
                                    projection_plot,
                                    corr_matrix_plot,
                                    plot_boxplot)
randomState = 17

dataDf, labels = load_dataset(dirs.dataset, fracPos=0.02, fracNeg=0.02, randomState=randomState)
dataDf = preproc(dataDf, verbose=True)

## Principal Components Analysis
# useful to reduce dataset dimensionality
compactDf = dimension_reduction(dataDf, keepComp=0)

## Plot Principal Components contributions
eigen_plot(dataDf, labels)

## Plot 3D data
print("\n3D Scatter plot")
plot_3d(compactDf.iloc[:, :3], labels)

## Custom pair plot
print("\nPair Projection plot")
projection_plot(compactDf.iloc[:,:10], labels)

## Correlation matrix plot
print("\nCorrelation matrix plot")
corr_matrix_plot(dataDf)

## Boxplot
print("\nBoxplot")
plot_boxplot(compactDf.iloc[:, :20], labels)

