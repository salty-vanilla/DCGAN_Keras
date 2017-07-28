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
    # (160, 128, 3) to (80, 64, 32)
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(32, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(0.2))

    # (80, 64, 32) to (40, 32, 64)
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(0.2))

    # (40, 32, 64) to (20, 16, 128)
    model.add(Conv2D(128, (3, 3), padding='same', input_shape=input_shape))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(0.2))

    # (20, 16, 128) to (10, 8, 256)
    model.add(Conv2D(256, (3, 3), padding='same', input_shape=input_shape))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(0.2))

    model.add(Flatten())

    # Input dim : (20480, )
    model.add(Dense(1024))
    model.add(LeakyReLU(0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    if plot_model:
        from keras.utils.vis_utils import plot_model as plot
        plot(model, to_file='model_generator.png', show_shapes=True)

    print("COMPLETE")
    return model
