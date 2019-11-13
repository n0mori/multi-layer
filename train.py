#!/usr/bin/env python3.7

import numpy as np
import math
import sys


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def eval_network(entrada, weights, biases, fn):
    hs = [entrada]
    result = entrada
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

def sigma_lminus(wl1, sl, hl, y):
    r = (np.transpose(np.copy(wl1)) * sl) * (hl * (1 - hl))
    return r


def train(dataset, arch, dims, reps):
    lr = 0.1

    ds = dataset[:, :dims]
    labels = dataset[:, dims:]

    thetas = [np.random.uniform(-1, 1, [v[1], v[0]]) for v in arch]
    biases = [np.random.uniform(-1, 1, [v[1]]) for v in arch]

    h = eval_network(ds[0], thetas, biases, sigmoid)

    for k in range(reps):
        print(f"\r{k+1}/{reps}", end='', flush=True)
        idx = np.random.randint(len(dataset))
        x, d = ds[idx], labels[idx]

        sigmas = [None for i in range(len(h))]

        for l in reversed(list(range(1, len(h)))):
            if l == len(h) - 1:
                sigmas[l] = sigma_l(d, h[l])
            else:
                sigmas[l] = sigma_lminus(thetas[l+1], sigmas[l+1], h[l], d)
        
        thetas[l] = thetas[l] - lr * sigmas[l] * np.transpose(np.copy(h[l-1]))
        biases[l] = biases[l] - lr * sigmas[l]

    print()
    return thetas, biases


# def output_nn(layers, biases):
#     i = 1
#     for l, b in zip(layers, biases):
#         print(f"camada{i}")
#         print(f"entrada {l.shape[1]}")
#         print(f"saida  {l.shape[0]}")
#         print('W')
#         for line in l:
#             print(line[0], end='')
#             for element in line[1:]:
#                 print(f' {element}', end='')
#             print()
#         print('b')
#         print(b[0][0], end='')
#         for element in b[1:]:
#             print(f' {element[0]}', end='')
#         print()
#         print('ativacao sigmoid')
#         print('--')

#         i += 1


def main():
    dataset = np.genfromtxt(f'train.csv', delimiter=',', skip_header=2)
    arch = [(784, 50), (50, 30), (30, 30), (30, 20), (20, 12), (12, 10)]

    layers, biases = train(dataset, arch, 784, 10)

    # output_nn(layers, biases)


if __name__ == "__main__":
    main()