#!/usr/bin/env python3.7

import numpy as np
import math
import sys

def sig(x):
    return 1.0 / (1.0 + math.exp(-x))


def activation(x, weights, bias):
    result = weights.dot(x)
    result = np.add(result, bias)
    return np.array([sig(x) for x in result])


def err(d, y, m):
    r = (2 * y[m] - 2 * d[m]) * (y[m] * (1 - y[m]))
    return r


def train(dataset, m, dims, reps):
    lr = 0.1

    ds = dataset[:, :dims]
    labels = dataset[:, dims:]

    theta = np.random.uniform(-1, 1, [m, dims])
    bias = np.random.uniform(-1, 1, [m])
    
    for k in range(reps):
        idx = np.random.randint(len(dataset))
        x, d = ds[idx], labels[idx]

        y = activation(x, theta, bias)

        for m in range(len(d)):
            for n in range(len(x)):
                theta[m][n] = theta[m][n] - lr * err(d, y, m) * x[n]
        
        for m in range(len(d)):
            bias[m] = bias[m] - lr * err(d, y, m)

    return theta, bias


def main():
    dataset = np.genfromtxt(f'ds{name}_cl{c}_dim{d}_treinamento.csv', delimiter=',', skip_header=2)
    t = np.genfromtxt(f'ds{name}_cl{c}_dim{d}_teste.csv', delimiter=',', skip_header=2)
    
    for k in [10, 100, 1000, 10000]:
        neuron, bias = train(dataset, c, d, k)


if __name__ == "__main__":
    main()