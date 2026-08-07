import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

##Day 1 - Neurons
'''
inputs = [1,2,3,2.5]
weights = [[0.2, 0.8, -0.5,1],[0.5, -0.91, 0.26,-0.5],[ -0.26, -0.27, 0.17, 0.87]]
bias = [2,3,1]

def neural_network(inputs, weights, bias):
    output = []
    for i in range(0,len(weights)):
        weighted_sum = float(np.dot(inputs, weights[i]) + bias[i])
        output.append(weighted_sum)
    return output

output = neural_network(inputs, weights, bias)
print(output)
'''
##DaY 2 - Dot Product
'''
a = [1,2,3]
b = [4,5,6]
a = np.array([a])
b = np.array([b]).T
print(np.dot(a, b))

inputs =[[1,2,3,2.5],
         [2.0,5.0,-1.0,2.0],
         [-1.5,2.7,3.3,-0.8]]
weights = [[0.2, 0.8, -0.5,1],
           [0.5, -0.91, 0.26,-0.5],
           [ -0.26, -0.27, 0.17, 0.87]]
biases = [2,3,0.5]
weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]
biases2 = [-1,2,-0.5]
layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
print(layer1_outputs)
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
print(layer2_outputs)

#########################################

X, y = spiral_data(samples=100, classes=3)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg")
plt.show()
'''
#Day 3 - Dense Layers

class Layer_Dense:
    def __init__(self,n_inputs, n_neurons):
        self.weights = 0.01*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2,3)

dense1.forward(X)
# print(dense1.output[:3])
##plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg")
##plt.show()
###Day 4 - Activation Functions

inputs = [0,2, -1, 3.3, -2.7, 1.1, 2.2, -100]


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

activation1 = Activation_ReLU()
activation1.forward(dense1.output)
print(activation1.output[:5])