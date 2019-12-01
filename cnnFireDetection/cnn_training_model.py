import tensorflow as tf
import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

import numpy as np
import time
import pickle



class cnnFireDetectionModel(object):
    def __init__(self):
        super().__init__()
        self.NAME = "Fire-vs-Tree-CNN-adam"
        self.tensorboard = TensorBoard(log_dir="logs\{}".format(self.NAME))
        self.opt = None
        
    def runModel (self, X, y):
        X = X/ 255.0
        y = np.array(y) # Need to move to step to other file

        #opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)

        model = Sequential()

        model.add(Conv2D(256, (3, 3), input_shape= X.shape[1:]))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))
        model.add(Activation('relu'))

        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

        model.fit(X, y, batch_size=32, epochs=15, validation_split=0.3, callbacks=[self.tensorboard])
        
        model.save('firedetector-adam-CNN.h5')
    
    def runModelOpt (self, X, y):
        X = X/ 255.0
        y = np.array(y) # Need to move to step to other file

        dense_layers = [0, 1, 2]
        layer_sizes = [32, 64, 128]
        conv_layers = [1, 2, 3]

        for dense_layer in dense_layers:
            for layer_size in layer_sizes:
                for conv_layer in conv_layers:
                    name = "Optimized-adam-model-{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))

                    model = Sequential()

                    model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
                    model.add(Activation('relu'))
                    model.add(MaxPooling2D(pool_size=(2, 2)))

                    for l in range(conv_layer-1):
                        model.add(Conv2D(layer_size, (3, 3)))
                        model.add(Activation('relu'))
                        model.add(MaxPooling2D(pool_size=(2, 2)))

                    model.add(Flatten())

                    for _ in range(dense_layer):
                        model.add(Dense(layer_size))
                        model.add(Activation('relu'))

                    model.add(Dense(1))
                    model.add(Activation('sigmoid'))

                    tensorboard = TensorBoard(log_dir="logs\{}".format(name))

                    model.compile(loss='binary_crossentropy',
                                optimizer='adam',
                                metrics=['accuracy'],
                                )

                    model.fit(X, y,
                            batch_size=32,
                            epochs=40,
                            validation_split=0.25,
                            callbacks=[tensorboard])
        
        model.save('firedetector-64x3-CNN.h5')