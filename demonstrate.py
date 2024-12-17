import main as network
import load_mnist

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


image = "data/images/a.png"
model = 'results/87_784-16-10/'


## Get image as vector and normalise

img = Image.open(image).convert('L')
img = np.array(img).flatten().reshape(784, 1) / 255


plt.imshow(img.reshape(28, 28), cmap='gray')
plt.title('Input Image')
plt.show()


nn = network.load_model(model)
print(f'Predicted number is: {nn.predict(img)[0]}')
