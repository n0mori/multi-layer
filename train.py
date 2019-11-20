#!/usr/bin/env python3.7

import numpy as np
import math
import sys


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def eval_network(entrada, weights, biases, fn):
    result = entrada.reshape(len(entrada), 1)
    hs = [result]
    for w, b in zip(weights, biases):
        result = eval_layer(result, w, b, fn)
        hs.append(result)
    return hs


def eval_layer(entrada, w, b, fn):
    result = np.matmul(w, entrada) + b
    vfn = np.vectorize(fn)
    return vfn(result)


def sigma_l(d, y):
    r = (2 * y - 2 * d) * (y * (1 - y))
    return r

def sigma_lminus(wl1, sl, hl):
    r = np.matmul(wl1.T, sl) * (hl * (1 - hl))
    return r


def train(dataset, arch, dims, reps):
    lr = 0.1

    ds = dataset[:, :dims]
    labels = dataset[:, dims:]

    thetas = [np.random.uniform(-1, 1, [v[1], v[0]]) for v in arch]
    biases = [np.random.uniform(-1, 1, [v[1], 1]) for v in arch]


    for k in range(reps):
        print(f"\r{k+1}/{reps}", end='', flush=True)
        idx = np.random.randint(len(dataset))
        x, d = ds[idx], labels[idx]
        d = d[np.newaxis].T

        h = eval_network(x, thetas, biases, sigmoid)
        sigmas = [None for i in range(len(h))]

        for l in reversed(list(range(1, len(h)))):
            if l == len(h) - 1:
                sigmas[l] = sigma_l(d, h[l])
            else:
                sigmas[l] = sigma_lminus(thetas[l], sigmas[l+1], h[l])

            thetas[l-1] = thetas[l-1] - (lr * np.matmul(sigmas[l], h[l-1].T))
            biases[l-1] = biases[l-1] - lr * sigmas[l]

    print()
    return thetas, biases


def output_nn(layers, biases):
    out = open("mnist_NN.txt", 'w+')
    i = 1
    for l, b in zip(layers, biases):
        print(f"camada{i}", file=out)
        print(f"entrada {l.shape[1]}", file=out)
        print(f"saida  {l.shape[0]}", file=out)
        print('W', file=out)
        for line in l:
            print(line[0], end='', file=out)
            for element in line[1:]:
                print(f' {element}', end='', file=out)
            print(file=out)
        print('b', file=out)
        print(b[0][0], end='', file=out)
        for element in b[1:]:
            print(f' {element[0]}', end='', file=out)
        print(file=out)
        print('ativacao sigmoid', file=out)
        print('--', file=out)

        i += 1

def test(layers, biases, testset):
    ds = testset[:, :784]
    labels = testset[:, 784:]

    erros = 0
    acertos = 0

    for x, l in zip(ds, labels):
        hs = eval_network(x, layers, biases, sigmoid)

        if np.argmax(hs[-1]) == np.argmax(l):
            acertos += 1
        else:
            erros += 1
    
    print('Resultados:', acertos, erros)


def main():
    dataset = np.genfromtxt(f'mnist_train.csv', delimiter=',', skip_header=2)
    testset = np.genfromtxt(f'mnist_test.csv', delimiter=',', skip_header=2)
    arch = [(784, 50), (50, 30), (30, 30), (30, 20), (20, 12), (12, 10)]

    layers, biases = train(dataset, arch, 784, 100000)
    test(layers, biases, testset)

    output_nn(layers, biases)


if __name__ == "__main__":
    main()