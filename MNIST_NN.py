print("NN V1.0", "\n")
import numpy as np

import MNIST_read
from Sigmoid import sigmoid




x, y, m, w, b = MNIST_read.read() #Reads x and y from the csv file and returns numpy arrays

print("\n", "x.sh:", x.shape, "y.sh:", y.shape, "w.sh:", w.shape, "b.sh:", b.shape, "\n")

z = np.dot(w, x.T) + b
z = np.sum(z, axis = 1) #compute z = sum(wx + b) for the weights
A = sigmoid(z)          #Sigmoid function on the weights
print("z.shape: ", z.shape)
print("z: ", (z))


print ("y: ", (y))
ylabel = y[0,0]       #assigns a 1 to the 5th value of the classifier as y= the character 5.
y = np.zeros(shape = (1,10))
y[0,4] = 1

print ("A:", (A))
print("y:", y)

cost = -(1 / m) * np.sum((y * np.log(A).T) + (1 - y) * np.log(1 - A).T) #cost function
print("\n", "cost: ", cost)

dz = A - y                                  #derivative z = derivative A with respect to Loss
dw = (1 / m) * np.dot(x.T, dz)              #derivative w = derivative w with respect to A
db = dz                                     #derivative for the bias = derivative z
print("\n", "dz:", dz.shape, "dw:", dw.shape, "db:", db.shape, "\n")

w = w - .001 * dw.T        # weight update
b = b - .001 * dz          #bias update
print("w.shape:", w.shape, "\n", "w:", w, "\n", "b.shape:", b.shape, "\n", "b:", b, "\n")


#Only currently running one iteration (m = 1) to test the code.

print("\n", "...Done")
