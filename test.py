import load_mnist
import main as network

import numpy as np


model = 'results/ideal/'

print("Loading data...")

data = load_mnist.Data('data/train-images.idx3-ubyte', 'data/train-labels.idx1-ubyte', 'data/t10k-images.idx3-ubyte', 'data/t10k-labels.idx1-ubyte')
images, labels = data.get_test_data()

inputs = np.array([img.flatten() / 255 for img in images]).T

print("Data loaded")

nn = network.load_model(model, inputs, labels)
nn.test()
