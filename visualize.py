from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

import os
import numpy as np
from src import modelinit
import matplotlib.pyplot as plt
from keras.preprocessing import image


def layer_to_visualize(layer):
    inputs = [K.learning_phase()] + model.inputs

    _convout1_f = K.function(inputs, [layer.output])

    def convout1_f(X):
        # The [0] is to disable the training phase flag
        return _convout1_f([0] + [X])

    img_to_visualize = image.load_img("C:\\Users\\Pavel\\Desktop\\8.jpg", target_size=(150, 150))
    img_to_visualize = image.img_to_array(img_to_visualize)
    img_to_visualize = np.expand_dims(img_to_visualize, axis=0)

    convolutions = convout1_f(img_to_visualize)
    convolutions = np.squeeze(convolutions)

    print('Shape of conv:', convolutions.shape)

    n = convolutions.shape[0]
    n = int(np.ceil(np.sqrt(n)))

    # Visualization of each filter of the layer
    fig = plt.figure(figsize=(12, 8))
    for i in range(len(convolutions) // 2):
        ax = fig.add_subplot(n, n, i+1)
        ax.imshow(convolutions[i], cmap='gray')

    fig.show()


model = modelinit.launch()
model.load_weights('weights.h5')

layer_to_visualize(model.layers[2])

a = input()