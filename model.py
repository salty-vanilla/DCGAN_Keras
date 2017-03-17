import cv2
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, BatchNormalization, Flatten, Reshape, Activation, LeakyReLU, ELU
from keras.models import Model, Sequential
from keras import initializations

def getDiscriminator(save_model=True):
    print("Building Discriminator ...")
    model = Sequential(name="discriminator")
    model.add(Convolution2D(64, 5, 5, border_mode='same', subsample=(2, 2), input_shape=(1, 28, 28), init=initNormal))
    model.add(LeakyReLU(0.2))
    model.add(Convolution2D(128, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(LeakyReLU(0.2))
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    if save_model:
        from keras.utils.visualize_util import plot
        plot(model, to_file='model_discriminator.png', show_shapes=True)

    return model

def getGenerator(save_model=True):
    print("Building Generator ...")
    model = Sequential(name="generator")
    model.add(Dense(input_dim=100, output_dim=(128 * 7 * 7), init=initNormal))
    model.add(Activation('relu'))
    model.add(Reshape((128, 7, 7)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(64, 5, 5, border_mode='same'))
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(1, 5, 5, border_mode='same'))
    model.add(Activation('tanh'))

    if save_model:
        from keras.utils.visualize_util import plot
        plot(model, to_file='model_generator.png', show_shapes=True)

    return model

def initNormal(shape, name=None):
    return initializations.normal(shape, scale=0.02, name=name)



