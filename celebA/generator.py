from keras.layers import (
    Dense,
    Conv2D,
    Conv2DTranspose,
    BatchNormalization,
    Reshape,
    Activation,
    LeakyReLU
)
from keras.models import Sequential
from keras import backend as K


def get_generator(input_dim, is_plot=False):
    print("Building Generator ...   ", end="")
    model = Sequential(name="generator")
    model.add(Dense(input_dim=input_dim, units=(256 * 10 * 8)))
    model.add(LeakyReLU(0.2))

    if K.image_dim_ordering() == 'th':
        model.add(Reshape((256, 10, 8)))
    else:
        model.add(Reshape((10, 8, 256)))

    # (10, 8, 256) to (20, 16, 128)
    model.add(Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    # (20, 16, 128) to (40, 32, 64)
    model.add(Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    # (40, 32, 64) to (80, 64, 32)
    model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    # (80, 64, 32) to (160, 128, 3)
    model.add(Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(3, (3, 3), padding='same'))
    model.add(Activation('tanh'))

    if is_plot:
        from keras.utils.vis_utils import plot_model
        plot_model(model, to_file='model_generator.png', show_shapes=True)

    print("COMPLETE")
    return model
