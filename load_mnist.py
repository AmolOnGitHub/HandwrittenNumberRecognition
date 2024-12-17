import numpy as np
import struct 
from array import array
from os.path import join
import matplotlib.pyplot as plt


def show_image(image, title='Image'):
    plt.imshow(image, cmap='gray')
    plt.title(title)


class Data():
    def __init__(self, training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath


    def get_image_labels(self, images_filepath, labels_filepath):
        with open(labels_filepath, 'rb') as file:
            # Magic number identifies file type, should be 2049 for label file
            # >II means big endian unsigned int

            magic, labels_size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(f'Magic number mismatch, expected 2049, got {magic}')
            
            labels = array("B", file.read())
            labels = np.array(labels)

        with open(images_filepath, 'rb') as file:
            # Magic should be 2051 for image file
            magic, images_size, rows, cols = struct.unpack(">IIII", file.read(16))

            if magic != 2051:
                raise ValueError(f'Magic number mismatch, expected 2051, got {magic}')
            
            image_data = array("B", file.read())

        if (labels_size != images_size):
            raise ValueError(f'Number of labels {labels_size} does not match number of images {images_size}')
        
        images = []

        for i in range(images_size):
            img = np.array(image_data[i * rows * cols: (i + 1) * rows * cols])
            img = img.reshape(rows, cols)
            images.append(img)

        return images, labels
    

    def get_training_data(self):
        return self.get_image_labels(self.training_images_filepath, self.training_labels_filepath)
    

    def get_test_data(self):
        return self.get_image_labels(self.test_images_filepath, self.test_labels_filepath)
    