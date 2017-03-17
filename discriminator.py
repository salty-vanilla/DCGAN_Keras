from keras.layers import (
    Convolution2D,
    LeakyReLU,
    Flatten,
    Dense,
    Activation
)
from keras.models import Sequential
from keras import initializations


def discriminator_mnist(input_shape, plot_model=False):
    def init_normal(shape, name=None):
        return initializations.normal(shape, scale=0.02, name=name)

    print("Building Discriminator ...   ", end="")
    model = Sequential(name="discriminator")
    model.add(Convolution2D(64, 5, 5, border_mode='same', subsample=(2, 2), input_shape=input_shape, init=init_normal))
    model.add(LeakyReLU(0.2))
    model.add(Convolution2D(128, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(LeakyReLU(0.2))
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    if plot_model:
        from keras.utils.visualize_util import plot
        plot(model, to_file='model_discriminator.png', show_shapes=True)

    print("COMPLETE")
    return model
