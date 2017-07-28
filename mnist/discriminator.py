from keras.layers import (
    Conv2D,
    LeakyReLU,
    Flatten,
    Dense,
    Activation,
)
from keras.models import Sequential


def discriminator_mnist(input_shape, plot_model=False):
    print("Building Discriminator ...   ", end="")
    model = Sequential(name="discriminator")
    model.add(Conv2D(64, (3, 3), padding='same', strides=(2, 2), input_shape=input_shape))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(128, (3, 3), padding='same', strides=(2, 2)))
    model.add(LeakyReLU(0.2))

    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    if plot_model:
        from keras.utils.vis_utils import plot_model as plot
        plot(model, to_file='model_discriminator.png', show_shapes=True)

    print("COMPLETE")
    return model
