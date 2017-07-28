# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from DCGAN import DCGAN
from celebA.generator import get_generator


input_dim = 100


def main():
    generator = get_generator(input_dim=input_dim)

    dcgan = DCGAN(input_dim, generator, None, param_path_generator="./generator.hdf5")
    dcgan.generate(dst_dir="generated", nb_generate=20)


if __name__ == "__main__":
    main()
