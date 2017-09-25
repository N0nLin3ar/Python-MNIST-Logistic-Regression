import numpy as np

def sigmoid(z):

    s = 1 / (1 + np.exp(-1 * z))


    return s