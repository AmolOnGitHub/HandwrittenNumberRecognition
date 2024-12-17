import load_mnist
import main as network

import numpy as np
import json

model = 'results/89_784-16-16-10/'

print("Loading data...")

data = load_mnist.Data('data/train-images.idx3-ubyte', 'data/train-labels.idx1-ubyte', 'data/t10k-images.idx3-ubyte', 'data/t10k-labels.idx1-ubyte')
images, labels = data.get_test_data()

inputs = np.array([img.flatten() / 255 for img in images]).T

print("Data loaded")

print("Loading model...")

with open(model + 'details.json', 'r') as f:
    details = json.load(f)

weights = []
biases = []

for i in range(len(details['structure']) - 1):
    weights.append(np.load(model + f'weight/{i}.npy'))
    biases.append(np.load(model + f'bias/{i}.npy'))

print("Model loaded\n")

nn = network.network(details['structure'], inputs, labels, weights=weights, biases=biases)
nn.test()


