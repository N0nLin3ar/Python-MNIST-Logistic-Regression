import numpy as np
import csv


def read():

    x_list = []
    y_list = []

    i_reader = csv.reader(open("mnist_images.csv"))
    l_reader = csv.reader(open("mnist_labels.csv"))


    m = 1     #This code is being run with only one iteration as a test of the code
    out = 10  #MNIST has 10 output classifiers

    for i in range(m):
        x_list.append(i_reader.__next__())  # images

    for i in range(m):
        y_list.append(l_reader.__next__())  # labels

    x = np.array(x_list, dtype=float)
    y = np.array(y_list, dtype=float)

    w = np.zeros((out, x.shape[1]))
    b = np.zeros(shape=(1, out))


    return (x, y, m, w, b)
