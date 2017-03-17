from DCGAN import DCGAN
from generator import generator_mnist


input_dim = 100


def main():
    generator = generator_mnist(input_dim=input_dim)

    dcgan = DCGAN(input_dim, generator, None, param_path_generator="./generator.hdf5")
    dcgan.generate(dst_dir="generated", nb_generate=20)


if __name__ == "__main__":
    main()
