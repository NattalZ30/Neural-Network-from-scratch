import numpy as np
import matplotlib.pyplot as plt

inputs = [1,2,3,2.5]
weights = [[0.2, 0.8, -0.5,1],[0.5, -0.91, 0.26,-0.5],[ -0.26, -0.27, 0.17, 0.87]]
bias = [2,3,1]

def neural_network(inputs, weights, bias):
    output = []
    for i in range(0,len(weights)):
        weighted_sum = np.dot(inputs, weights[i]) + bias[i]
        output.append(float(weighted_sum))
    return output

output = neural_network(inputs, weights, bias)
print(output)