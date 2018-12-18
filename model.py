from __future__ import absolute_import
from __future__ import print_function
import numpy as np

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Flatten, Dense, Conv3D, MaxPool3D,MaxPooling2D, PReLU


def _3d_cnn_model(input_shape, num_classes):
    # Define Model
    inputs = Input(shape=input_shape, name="input-layer")

    # Conv 1
    X = Conv3D(filters=16, kernel_size=(3, 1, 5), strides=(1, 1, 1), name="conv1-1")(inputs)
    X = PReLU(name="activation1-1")(X)
    X = Conv3D(filters=16, kernel_size=(3, 9, 1), strides=(1, 2, 1), name="conv1-2")(X)
    X = PReLU(name="activation1-2")(X)
    X = MaxPool3D(pool_size=(1, 1, 2), strides=(1, 1, 2), padding="valid", name="pool-1")(X)
    # X = Dropout(0.2)(X)

    # Conv 2
    X = Conv3D(filters=32, kernel_size=(3, 1, 4), strides=(1, 1, 1), name="conv2-1")(X)
    X = PReLU(name="activation2-1")(X)
    X = Conv3D(filters=32, kernel_size=(3, 8, 1), strides=(1, 2, 1), name="conv2-2")(X)
    X = PReLU(name="activation2-2")(X)
    X = MaxPool3D(pool_size=(1, 1, 2), strides=(1, 1, 2), padding="valid", name="pool-2")(X)
    # X = Dropout(0.2)(X)

    # Conv 3
    X = Conv3D(filters=64, kernel_size=(3, 1, 3), strides=(1, 1, 1), name="conv3-1")(X)
    X = PReLU(name="activation3-1")(X)
    X = Conv3D(filters=64, kernel_size=(3, 7, 1), strides=(1, 1, 1), name="conv3-2")(X)
    X = PReLU(name="activation3-2")(X)
    # X = Dropout(0.2)(X)

    # Conv 4
    X = Conv3D(filters=128, kernel_size=(3, 1, 3), strides=(1, 1, 1), name="conv4-1")(X)
    X = PReLU(name="activation4-1")(X)
    X = Conv3D(filters=128, kernel_size=(3, 7, 1), strides=(1, 1, 1), name="conv4-2")(X)
    X = PReLU(name="activation4-2")(X)
    # X = Dropout(0.2)(X)

    # Flaten
    X = Flatten()(X)

    # FC
    X = Dense(units=128, name="fc", activation='relu')(X)

    # Final Activation
    X = Dense(units=num_classes, activation='softmax', name="ac_softmax")(X)
    model = Model(inputs=inputs, outputs=X)

    return model



