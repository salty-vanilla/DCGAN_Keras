from keras.layers import (
    Convolution2D,
    LeakyReLU,
    Flatten,
    Dense,
    Activation
)
from keras.models import Sequential


def discriminator_mnist(input_shape, plot_model=False):
    print("Building Discriminator ...   ", end="")
    model = Sequential(name="discriminator")
    model.add(Convolution2D(64, (5, 5), padding='same', strides=(2, 2), input_shape=input_shape))
    model.add(LeakyReLU(0.2))
    model.add(Convolution2D(128, (5, 5), padding='same', strides=(2, 2)))
    model.add(LeakyReLU(0.2))
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    if plot_model:
        from keras.utils.vis_utils import plot_model as plot
        plot(model, to_file='model_generator.png', show_shapes=True)

    print("COMPLETE")
    return model
