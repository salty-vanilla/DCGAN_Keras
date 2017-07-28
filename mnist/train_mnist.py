# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import sys
import argparse
try:
    import cPickle as pickle
except:
    import pickle
import gzip
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.getcwd())
from keras.optimizers import Adam
from DCGAN import DCGAN
from mnist.discriminator import discriminator_mnist
from mnist.generator import generator_mnist
from keras import backend as K


width = height = 28
channel = 1


def data_init():
    print("Loading MNIST ...    ", end="")
    f = gzip.open('./mnist/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

    real_images, _ = train_set

    if K.image_dim_ordering() == 'th':
        real_images = (real_images.reshape(real_images.shape[0], channel, height, width).astype("float32") - 0.5) / 0.5
    else:
        real_images = (real_images.reshape(real_images.shape[0], height, width, channel).astype("float32") - 0.5) / 0.5
    print("COMPLETE")
    return real_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_dim', '-ld', type=int, default=100)
    parser.add_argument('--batch_size', '-bs', type=int, default=300)
    parser.add_argument('--nb_epoch', '-e', type=int, default=50)
    parser.add_argument('--visualize_steps', '-vs', type=int, default=1)
    parser.add_argument('--save_steps', '-ss', type=int, default=1)
    parser.add_argument('--result_dir', '-rd', type=str, default="./mnist/result")
    parser.add_argument('--param_dir', '-pd', type=str, default="./mnist/params")

    args = parser.parse_args()

    input_dim = args.latent_dim
    batch_size = args.batch_size
    nb_epoch = args.nb_epoch
    visualize_steps = args.visualize_steps
    save_steps = args.save_steps
    result_dir = args.result_dir
    param_dir = args.param_dir

    if K.image_dim_ordering() == 'th':
        input_shape_d = (channel, height, width)
    else:
        input_shape_d = (height, width, channel)

    real_images = data_init()
    generator = generator_mnist(input_dim=input_dim)
    # output model to json
    open(os.path.join(param_dir, 'generator.json'), 'w').write(generator.to_json())
    discriminator = discriminator_mnist(input_shape=input_shape_d)

    dcgan = DCGAN(input_dim, generator, discriminator)

    opt_d = Adam(lr=1e-4, beta_1=0.1)
    opt_d_params = {'opt': opt_d,
                    'loss': 'binary_crossentropy',
                    'metrics': ['accuracy']}

    opt_g = Adam(lr=2e-3, beta_1=0.5)
    opt_g_params = {'opt': opt_g,
                    'loss': 'binary_crossentropy',
                    'metrics': ['accuracy']}

    dcgan.build(opt_g_params, opt_d_params)

    dcgan.fit(real_images, batch_size=batch_size, nb_epoch=nb_epoch,
              param_dir=param_dir, log_csv_path=os.path.join(result_dir, "result.csv"),
              save_steps=save_steps, visualize_steps=visualize_steps)


if __name__ == "__main__":
    main()
