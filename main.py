import load_mnist

import numpy as np
from scipy.ndimage import shift, rotate, zoom
import os
import json


NABLA = 0.1
BATCH_SIZE = 10
NORMALISATION = 'RELU'

def augment_data(images, image_size=(28, 28)):
    ## Augment the data by shifting, rotating and zooming the images

    print("\nAugmenting data...")
    augmented_images = []
    
    for img in images:
        img = img.reshape(image_size)

        # Random shifts

        shifted_img = shift(img, shift=(np.random.randint(-2, 3), np.random.randint(-2, 3)), mode='constant')
        rotated_img = rotate(shifted_img, angle=np.random.uniform(-15, 15), reshape=False, mode='constant')
        zoom_factor = np.random.uniform(0.9, 1.1)
        zoomed_img = zoom(rotated_img, zoom_factor)


        # Fit back to 28x28

        h, w = zoomed_img.shape
        cropped_img = np.zeros(image_size)
        
        min_h = min(h, image_size[0])
        min_w = min(w, image_size[1])

        start_h = (h - min_h) // 2
        start_w = (w - min_w) // 2

        cropped_img[:min_h, :min_w] = zoomed_img[start_h:start_h + min_h, start_w:start_w + min_w]

        augmented_images.append(cropped_img.flatten())


    print("Data augmented\n")
    return np.array(augmented_images)


def load_model(file_path, inputs=None, labels=None):
    ## Load a model from the given file path

    print("\nLoading model...")

    with open(file_path + 'details.json', 'r') as f:
        details = json.load(f)

    weights = []
    biases = []

    for i in range(len(details['structure']) - 1):
        weights.append(np.load(file_path + f'weight/{i}.npy'))
        biases.append(np.load(file_path + f'bias/{i}.npy'))

    print("Model loaded\n")

    return network(details['structure'], inputs, labels, weights=weights, biases=biases)


class network():
    def __init__(self, structure, inputs, labels, weights=None, biases=None):
        ## Initialise the network with the given structure, weights and biases

        self.structure = structure

        if weights is None:
            self.weights = [np.random.rand(structure[i], structure[i - 1]) - 0.5 for i in range(1, len(structure))]
        else: 
            self.weights = weights

        if biases is None:
            self.biases = [np.random.rand(structure[i], 1) for i in range(1, len(structure))]
        else:
            self.biases = biases

        if inputs is not None and labels is not None: 
            self.inputs = inputs
            self.m = inputs.shape[1]
            self.labels = labels


    def ReLU(self, x):
        return np.maximum(0, x)
    

    def ReLU_derivative(self, x):
        return x > 0
    

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)


    def forward_prop(self, X):
        ## Takes in matrix X of shape (n, m) where n is the number of features and m is the number of samples
        ## n here should match the number of input neurons, i.e. structure[0]

        a = X
        A = [a]
        Z = [a]
        
        for i in range(len(self.weights)):
            Wi = self.weights[i]
            bi = self.biases[i]

            z = Wi.dot(a) + bi

            if (i < len(self.weights) - 1):
                adash = self.ReLU(z)
            else:
                adash = self.softmax(z)
            
            a = adash
            A.append(a)
            Z.append(z)

        
        return Z, A
    

    def backward_prop(self, Z, A, Y):
        ## Calculate the derivative of the cost function with respect to the weights and biases

        hot = np.zeros((Y.max() + 1, self.m))
        hot[Y, np.arange(self.m)] = 1

        dW = []
        dB = []

        dz = A[-1] - hot

        for i in range(len(self.weights) - 1, -1, -1):
            dw = 1 / self.m * dz.dot(A[i].T)
            db = 1 / self.m * np.sum(dz)

            if (i > 0):
                dzdash = self.weights[i].T.dot(dz) * self.ReLU_derivative(Z[i])
                dz = dzdash

            dW.insert(0, dw)
            dB.insert(0, db)


        # Update weights and biases

        for i in range(len(self.weights)):
            self.weights[i] -= NABLA * dW[i]
            self.biases[i] -= NABLA * dB[i]


    def get_prediction(self, probabilities):
        ## Get the prediction from the probabilities using argmax

        return np.argmax(probabilities, 0);


    def get_accuracy(self, predictions, Y):
        ## Get the accuracy of the model
        
        return np.sum(predictions == Y) / self.m


    def gradient_descent(self, iterations, save=False):
        ## Perform gradient descent on the network

        for i in range(1, iterations + 1):
            # Shuffle the data

            shuffled_indices = np.random.permutation(self.m)
            self.inputs = self.inputs[:, shuffled_indices]
            self.labels = self.labels[shuffled_indices]

            
            # Forward propagate to get activations, then back propagate to get gradient and update weights and biases

            Z, A = self.forward_prop(self.inputs)
            self.backward_prop(Z, A, self.labels)


            if (i % BATCH_SIZE == 0):
                predictions = self.get_prediction(A[-1])
                accuracy = self.get_accuracy(predictions, self.labels)
                print(f"{i} | Accuracy: {accuracy * 100:.2f}%")


            # Augment data every quarter of the iterations

            if i % (iterations / 4) == 0:
                aug_data = augment_data(self.inputs.T).T
                self.inputs = np.concatenate((self.inputs, aug_data), axis=1)
                self.labels = np.concatenate((self.labels, self.labels), axis=0)
                self.m = self.inputs.shape[1]


        if save:
            # Save model

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
        ## Test the model on the the given inputs

        _, A = self.forward_prop(self.inputs)
        predictions = self.get_prediction(A[-1])
        accuracy = self.get_accuracy(predictions, self.labels)
        print(f"{predictions} {self.labels} | {accuracy * 100:.2f}%")


    def predict(self, X):
        ## Given an input X, predict the output

        _, A = self.forward_prop(X)
        return A[-1], self.get_prediction(A[-1])



if __name__ == '__main__':
    data = load_mnist.Data('data/train-images.idx3-ubyte', 'data/train-labels.idx1-ubyte', 'data/t10k-images.idx3-ubyte', 'data/t10k-labels.idx1-ubyte')

    im, lb = data.get_training_data()
    inp = np.array([img.flatten() / 255 for img in im]).T

    nn = network([784, 100, 32, 16, 10], inp, lb)
    nn.gradient_descent(10000, save=True)
