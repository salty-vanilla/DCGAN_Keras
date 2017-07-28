# -*- coding: utf-8 -*-
import os
import sys
import argparse
sys.path.append(os.getcwd())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
from DCGAN import DCGAN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_dim', '-ld', type=int, default=100)
    parser.add_argument('--model_path', '-mp', type=str, default='./mnist/params/generator.json')
    parser.add_argument('--param_path', '-pp', type=str, default='./mnist/params/generator_{}.hdf5'.format(10))
    parser.add_argument('--nb_generate', '-n', type=int, default=50)
    parser.add_argument('--result_dir', '-rd', type=str, default="./mnist/result/generated")

    args = parser.parse_args()

    input_dim = args.latent_dim
    model_path = args.model_path
    param_path = args.param_path
    nb_generate = args.nb_generate
    result_dir = args.result_dir

    generator = keras.models.model_from_json(open(model_path).read())
    generator.load_weights(param_path)

    dcgan = DCGAN(input_dim, generator)
    dcgan.generate(result_dir, nb_generate)


if __name__ == "__main__":
    main()
