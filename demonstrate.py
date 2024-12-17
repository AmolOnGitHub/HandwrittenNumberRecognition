import main as network
import load_mnist

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random

image = "data/images/test.png"
model = 'results/93_784-16-32-16-10/'


images, labels = load_mnist.Data('data/train-images.idx3-ubyte', 'data/train-labels.idx1-ubyte', 'data/t10k-images.idx3-ubyte', 'data/t10k-labels.idx1-ubyte').get_test_data()

index = random.randint(0, len(images) - 1)
img = images[index].flatten().reshape(784, 1) / 255

plt.imshow(img.reshape(28, 28), cmap='gray')
plt.title(f'Image: {labels[index]}')
plt.show()


nn = network.load_model(model)
print(f'Predicted number is: {nn.predict(img)[0]}')
