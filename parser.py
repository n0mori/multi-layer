#!/usr/bin/env python3.7

import numpy as np
import math


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def eval_layer(entrada, w, b, fn):
    result = np.matmul(w, entrada) + b
    vfn = np.vectorize(fn)
    return vfn(result)


def parse(name):
    weights = []
    biases = []
    activations = []
    mode = ""
    entrada = 0
    saida = 0
    i = 0
    w = np.zeros((saida, entrada))
    b = np.zeros(1)

    with open(name) as f:
        for line in f:
            if mode == '':
                if line[0] == '-':
                    continue
                if line[0] == 'c':
                    continue
                if line[0] == 'e':
                    entrada = int(line.split(" ")[1])
                if line[0] == 's':
                    saida = int(line.split("  ")[1])
                if line[0] == 'W':
                    w = np.zeros((saida, entrada))
                    i = 0
                    mode = 'w'
            elif mode == 'w':
                if line[0] == 'b':
                    mode = 'b'
                    i = 0
                    weights.append(w)
                    b = np.zeros(saida)
                    continue
                values = [float(x) for x in line.split(" ")]
                for k, v in enumerate(values):
                    w[i][k] = v
                i += 1
            elif mode == 'b':
                if line[0] == 'a':
                    activations.append(line.strip().split(" ")[1])
                    mode = ''
                    i = 0
                    biases.append(b)
                    continue
                for k, v in enumerate([float(x) for x in line.split(" ")]):
                    b[k] = v

    return weights, biases, activations

def test(dataset, labels, weights, biases, activations):
    fns = {"sigmoid": sigmoid}

    predictions = []

    for x, l in zip(dataset, labels):
        result = x
        for w, b, a in zip(weights, biases, activations):
            result = eval_layer(result, w, b, fns[a])
        predictions.append(result)
        
    for m in range(labels.shape[1]):
        tp = fp = tn = fn = 0
        for l, p in zip(labels, predictions):
            pred = 1 if np.argmax(p) == m else 0
            if labels.shape[1] == 1:
                pred = 1 if p[0] > 0.5 else 0
            real = l[m]

            if pred == 1 and real == 1:
                tp += 1
            elif pred == 1 and real == 0:
                fp += 1
            elif pred == 0 and real == 0:
                tn += 1
            elif pred == 0 and real == 1:
                fn += 1

        prec = round(tp / float(tp + fp), 2) if (tp + fp) > 0 else "N/A"
        recall = round(tp / float(tp + fn), 2) if (tp + fn) > 0 else "N/A"
        print(m, prec,recall, sep=' & ', end=' \\\\\n')
    print()


def main(name, dims):
    weights, biases, activations = parse(name + "_NN.txt")

    raw_dataset = np.genfromtxt(name + "_test.csv", delimiter=',', skip_header=2)

    dataset = raw_dataset[:,:dims]
    labels = raw_dataset[:,dims:]

    print(name)
    print("class,pred,recall")
    test(dataset, labels, weights, biases, activations)



if __name__ == "__main__":
    main("LS", 2)
    main("MDML", 5)
    main("NL", 2)
    main("XOR", 2)