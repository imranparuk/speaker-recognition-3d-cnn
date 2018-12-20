from __future__ import absolute_import
from __future__ import print_function

from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.losses import categorical_crossentropy

from src.keras.dataset import DataGenerator
from src.keras.model import _3d_cnn_model

if __name__ == "__main__":

    num_classes = 10
    input_shape = (20, 80, 40, 1)
    epochs = 1

    train_iter = DataGenerator('dummy_dir', 10)

    model = _3d_cnn_model(input_shape, num_classes)

    opt = Adam()
    loss = categorical_crossentropy

    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=['accuracy'])

    model.fit_generator(train_iter,
                        epochs=epochs,
                        callbacks=None,
                        )


