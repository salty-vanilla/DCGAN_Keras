from keras.layers import (
    Dense,
    Convolution2D,
    UpSampling2D,
    Reshape,
    Activation
)
from keras.models import Sequential
from keras import backend as K


def generator_mnist(input_dim, plot_model=False):
    print("Building Generator ...   ", end="")
    model = Sequential(name="generator")
    model.add(Dense(input_dim=input_dim, units=(128 * 7 * 7)))
    model.add(Activation('relu'))

    if K.image_dim_ordering() == 'th':
        model.add(Reshape((128, 7, 7)))
    else:
        model.add(Reshape((7, 7, 128)))

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(64, (5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))

    if plot_model:
        from keras.utils.vis_utils import plot_model as plot
        plot(model, to_file='model_generator.png', show_shapes=True)

    print("COMPLETE")
    return model
