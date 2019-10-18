#!/usr/bin/env python3.7

import numpy as np
import math

def sigmoid(x):
    pass

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
                    w[k] = v

    return weights, biases, activations


def main(name):
    weights, biases, activations = parse(name)
    
    for w, b, a in zip(weights, biases, activations):
        print(w)
        print(b)
        print(a)



if __name__ == "__main__":
    main("NL_NN.txt")