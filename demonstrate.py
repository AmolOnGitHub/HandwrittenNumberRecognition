import main as network

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


image = "data/images/6.png"
model = 'results/ideal/'


img = Image.open(image).convert('L')
img = img.resize((28, 28))
img = np.array(img).flatten().reshape(784, 1) / 255


plt.imshow(img.reshape(28, 28), cmap='gray')
plt.show()


nn = network.load_model(model)
probs, prediction = nn.predict(img)

print(f'Predicted number is: {prediction[0]} with a probability of {np.max(probs)}')

if np.max(probs) < 0.75:
    probs[np.argmax(probs)] = 0
    print(f"Next guess is: {np.argmax(probs)} with a probability of {np.max(probs)}")
