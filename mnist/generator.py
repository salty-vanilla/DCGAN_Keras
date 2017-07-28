from keras.layers import (
    Dense,
    Conv2D,
    Conv2DTranspose,
    Reshape,
    Activation,
    BatchNormalization,
    LeakyReLU,
    UpSampling2D
)
from keras.models import Sequential
from keras import backend as K


def generator_mnist(input_dim, is_plot=False):
    print("Building Generator ...   ", end="")
    model = Sequential(name="generator")
    model.add(Dense(input_dim=input_dim, units=(128 * 7 * 7)))
    model.add(LeakyReLU(0.2))

    if K.image_dim_ordering() == 'th':
        model.add(Reshape((128, 7, 7)))
    else:
        model.add(Reshape((7, 7, 128)))

    # model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same'))
    # model.add(BatchNormalization())
    # model.add(LeakyReLU(0.2))
    # model.add(Conv2D(32, (3, 3), padding='same'))
    # model.add(BatchNormalization())
    # model.add(LeakyReLU(0.2))
    #
    # model.add(Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same'))
    # model.add(BatchNormalization())
    # model.add(LeakyReLU(0.2))
    # model.add(Conv2D(1, (3, 3), padding='same'))
    # model.add(Activation('tanh'))

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))

    if is_plot:
        from keras.utils.vis_utils import plot_model
        plot_model(model, to_file='model_generator.png', show_shapes=True)

    print("COMPLETE")
    return model
