import numpy                as np
import pandas               as pd

def show_class_splits(labels):
    entriesPos = np.sum(labels == +1)
    entriesNeg = np.sum(labels == -1)
    total = len(labels)

    print("Positive class: {:3.2f} %, {} entries ".format(entriesPos/total*100, entriesPos))
    print("Negative class: {:3.2f} %, {} entries ".format(entriesNeg/total*100, entriesNeg))
    print("Total:          {} entries".format(total))

    return

