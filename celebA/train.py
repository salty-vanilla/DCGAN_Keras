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
from DataGenerator import DataGenerator
from celebA.discriminator import get_discriminator
from celebA.generator import get_generator
from keras import backend as K


target_size = (128, 160)
color_mode = 'rgb'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', '-id', type=str, default='./dataset/celeba')
    parser.add_argument('--latent_dim', '-ld', type=int, default=100)
    parser.add_argument('--batch_size', '-bs', type=int, default=32)
    parser.add_argument('--nb_epoch', '-e', type=int, default=50)
    parser.add_argument('--visualize_steps', '-vs', type=int, default=1)
    parser.add_argument('--save_steps', '-ss', type=int, default=1)
    parser.add_argument('--result_dir', '-rd', type=str, default="./celebA/result")
    parser.add_argument('--param_dir', '-pd', type=str, default="./celebA/params")

    args = parser.parse_args()

    image_dir = args.image_dir
    input_dim = args.latent_dim
    batch_size = args.batch_size
    nb_epoch = args.nb_epoch
    visualize_steps = args.visualize_steps
    save_steps = args.save_steps
    result_dir = args.result_dir
    param_dir = args.param_dir

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(param_dir, exist_ok=True)

    width, height = target_size
    channel = 3 if color_mode == 'rgb' else 1
    if K.image_dim_ordering() == 'th':
        input_shape_d = (channel, height, width)
    else:
        input_shape_d = (height, width, channel)

    data_gen = DataGenerator(image_dir, target_size, color_mode, nb_samples=30000)
    generator = get_generator(input_dim=input_dim)
    # output model to json
    open(os.path.join(param_dir, 'generator.json'), 'w').write(generator.to_json())
    discriminator = get_discriminator(input_shape=input_shape_d, is_plot=True)

    dcgan = DCGAN(input_dim, generator, discriminator)

    opt_d = Adam(lr=0.0002, beta_1=0.1)
    opt_d_params = {'opt': opt_d,
                    'loss': 'binary_crossentropy',
                    'metrics': ['accuracy']}

    opt_g = Adam(lr=0.002, beta_1=0.5)
    opt_g_params = {'opt': opt_g,
                    'loss': 'binary_crossentropy',
                    'metrics': ['accuracy']}

    dcgan.build(opt_g_params=opt_g_params,
                opt_d_params=opt_d_params)

    dcgan.fit_generator(data_gen.flow(batch_size), samples_per_epoch=data_gen.data_num,
                        batch_size=batch_size, nb_epoch=nb_epoch,
                        param_dir=param_dir, log_csv_path=os.path.join(result_dir, "result.csv"),
                        is_separate=True, is_noisy_label=False,
                        save_steps=save_steps, visualize_steps=visualize_steps)


if __name__ == "__main__":
    main()