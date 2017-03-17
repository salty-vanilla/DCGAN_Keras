# -*- coding: utf-8 -*-
from __future__ import print_function
from keras.optimizers import Adam
try:
    import cPickle as pickle
except:
    import pickle
import gzip
from DCGAN import DCGAN
from discriminator import discriminator_mnist
from generator import generator_mnist
from keras import backend as K


nb_epoch = 1000
batch_size = 300
visualize_duration = 1
input_dim = 100
width = height = 28
channel = 1


def data_init():
    print("Loading MNIST ...    ", end="")
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

    real_images, _ = train_set
    real_images = (real_images.reshape(real_images.shape[0], 1, 28, 28).astype("float32") - 0.5) / 0.5

    print("COMPLETE")
    return real_images


def main():
    if K.image_dim_ordering() == 'th':
        input_shape_d = (channel, height, width)
    else:
        input_shape_d = (height, width, channel)

    real_images = data_init()
    generator = generator_mnist(input_dim=input_dim)
    discriminator = discriminator_mnist(input_shape=input_shape_d)

    dcgan = DCGAN(input_dim, generator, discriminator)

    opt_d = Adam(lr=1e-5, beta_1=0.1)
    opt_d_params = {'opt': opt_d,
                    'loss': 'binary_crossentropy',
                    'metrics': ['accuracy']}

    opt_g = Adam(lr=2e-4, beta_1=0.5)
    opt_g_params = {'opt': opt_g,
                    'loss': 'binary_crossentropy',
                    'metrics': ['accuracy']}

    dcgan.build(opt_g_params, opt_d_params)

    dcgan.fit(real_images, batch_size=batch_size, nb_epoch=nb_epoch, log_csv_path="result.csv")


if __name__ == "__main__":
    main()
