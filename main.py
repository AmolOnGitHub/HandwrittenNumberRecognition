import load_mnist

import numpy as np
import random
import os
import json


NABLA = 0.2
BATCH_SIZE = 10
NORMALISATION = 'RELU'


class network():
    def __init__(self, structure, inputs, labels, weights=None, biases=None):
        self.structure = structure

        if weights is None:
            self.weights = [np.random.rand(structure[i], structure[i - 1]) - 0.5 for i in range(1, len(structure))]
        else: 
            self.weights = weights

        if biases is None:
            self.biases = [np.random.rand(structure[i], 1) for i in range(1, len(structure))]
        else:
            self.biases = biases

        self.inputs = inputs
        self.labels = labels
        self.m = inputs.shape[1]


    def ReLU(self, x):
        return np.maximum(0, x)
    

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    

    def ReLU_derivative(self, x):
        return x > 0
    

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)


    def normalise(self, x):
        if NORMALISATION == 'RELU':
            return self.ReLU(x)
        else:
            return self.sigmoid(x)


    def normalise_derivative(self, x):
        if NORMALISATION == 'RELU':
            return self.ReLU_derivative(x)
        else:
            return self.sigmoid_derivative(x)


    def forward_prop(self, X):
        # Takes in matrix X of shape (n, m) where n is the number of features and m is the number of samples
        # n here should match the number of input neurons, i.e. structure[0]

        a = X
        A = [a]
        Z = [a]
        
        for i in range(len(self.weights)):
            Wi = self.weights[i]
            bi = self.biases[i]

            z = Wi.dot(a) + bi
            if (i < len(self.weights) - 1):
                adash = self.normalise(z)
            else:
                adash = self.softmax(z)
            
            a = adash
            A.append(a)
            Z.append(z)

        
        return Z, A
    

    def backward_prop(self, Z, A, Y):
        # Calculate the derivative of the cost function with respect to the weights and biases

        hot = np.zeros((Y.max() + 1, self.m))
        hot[Y, np.arange(self.m)] = 1

        dW = []
        dB = []

        dz = A[-1] - hot

        for i in range(len(self.weights) - 1, -1, -1):
            dw = 1 / self.m * dz.dot(A[i].T)
            db = 1 / self.m * np.sum(dz)

            if (i > 0):
                dzdash = self.weights[i].T.dot(dz) * self.normalise_derivative(Z[i])
                dz = dzdash

            dW.insert(0, dw)
            dB.insert(0, db)


        # Update weights and biases

        for i in range(len(self.weights)):
            self.weights[i] -= NABLA * dW[i]
            self.biases[i] -= NABLA * dB[i]


    def get_prediction(self, probabilities):
        return np.argmax(probabilities, 0);


    def get_accuracy(self, predictions, Y):
        return np.sum(predictions == Y) / self.m


    def gradient_descent(self, iterations, save=False):
        for i in range(iterations):
            Z, A = self.forward_prop(self.inputs)
            self.backward_prop(Z, A, self.labels)

            if (i % BATCH_SIZE == 0):
                predictions = self.get_prediction(A[-1])
                accuracy = self.get_accuracy(predictions, self.labels)
                print(f"{i} | {predictions} {self.labels} {accuracy}")


        if save:
            source_dest = f'results/{accuracy * 100:.0f}_{'-'.join(map(str, self.structure))}/'

            if not os.path.exists(source_dest):
                os.makedirs(source_dest + '/weight')
                os.makedirs(source_dest + '/bias')

            details = {
                "accuracy": accuracy,
                "structure": self.structure,
                "normalisation": NORMALISATION,
                "NABLA": NABLA,
                "iterations": iterations,
                "batch_size": BATCH_SIZE
            }
            with open(source_dest + 'details.json', 'w') as f:
                json.dump(details, f, indent=4)


            for i, weight in enumerate(self.weights):
                np.save(f'{source_dest}/weight/{i}.npy', weight)
            for i, bias in enumerate(self.biases):
                np.save(f'{source_dest}/bias/{i}.npy', bias)

            print(f"\nSaved to {source_dest}\n")

        return self.weights, self.biases
    

    def test(self):
        _, A = self.forward_prop(self.inputs)
        predictions = self.get_prediction(A[-1])
        accuracy = self.get_accuracy(predictions, self.labels)
        print(f"{predictions} {self.labels} | {accuracy * 100:.2f}%")



if __name__ == '__main__':
    data = load_mnist.Data('data/train-images.idx3-ubyte', 'data/train-labels.idx1-ubyte', 'data/t10k-images.idx3-ubyte', 'data/t10k-labels.idx1-ubyte')

    im, lb = data.get_training_data()
    inp = np.array([img.flatten() / 255 for img in im]).T

    nn = network([784, 16, 16, 10], inp, lb)
    nn.gradient_descent(500, save=True)
