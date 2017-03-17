from keras.layers import (
    Dense,
    Convolution2D,
    UpSampling2D,
    Reshape,
    Activation
)
from keras.models import Sequential
from keras import initializations
from keras import backend as K


def generator_mnist(input_dim, plot_model=False):
    def init_normal(shape, name=None):
        return initializations.normal(shape, scale=0.02, name=name)

    print("Building Generator ...")
    model = Sequential(name="generator")
    model.add(Dense(input_dim=input_dim, output_dim=(128 * 7 * 7), init=init_normal))
    model.add(Activation('relu'))

    if K.image_dim_ordering() == 'th':
        model.add(Reshape((128, 7, 7)))
    else:
        model.add(Reshape((7, 7, 128)))

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(64, 5, 5, border_mode='same'))
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(1, 5, 5, border_mode='same'))
    model.add(Activation('tanh'))

    if plot_model:
        from keras.utils.visualize_util import plot
        plot(model, to_file='model_generator.png', show_shapes=True)

    return model
