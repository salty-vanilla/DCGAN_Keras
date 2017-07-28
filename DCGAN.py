from keras.layers import Input
from keras.models import Model
from keras import backend as K
import numpy as np
import sys
import csv
import os
import cv2
import math


class DCGAN:

    def __init__(self, input_dim, generator, discriminator=None,
                 param_path_generator=None, param_path_discriminator=None):
        """
        コンストラクタ
        :param input_dim: generatorの入力次元数
        :param generator:
        :param discriminator:
        :param param_path_generator:
        :param param_path_discriminator:
        """
        self.input_dim = input_dim
        self.generator = generator
        self.discriminator = discriminator
        self.dcgan = None
        self.on_training = False
        self.param_path_generator = param_path_generator
        self.param_path_discriminator = param_path_discriminator

    def build(self, opt_g_params, opt_d_params):
        """
        モデルをコンパイルするための関数
        :param opt_g_params: generatorに関するパラメータ。辞書クラス
        :param opt_d_params: discriminatorに関するパラメータ。辞書クラス
        :return:

        辞書はそれぞれ以下のキーを持つ
        opt: kerasの最適化クラス
        loss: loss関数
        metrics: metrics
        """

        self.discriminator.compile(optimizer=opt_d_params['opt'],
                                   loss=opt_d_params['loss'],
                                   metrics=opt_d_params['metrics'])
        self.discriminator.trainable = False

        inputs = Input(shape=(self.input_dim, ))
        generated = self.generator(inputs)
        discriminated = self.discriminator(generated)
        self.dcgan = Model(inputs=inputs, outputs=discriminated)
        self.dcgan.compile(optimizer=opt_g_params['opt'],
                           loss=opt_g_params['loss'],
                           metrics=opt_g_params['metrics'])

    def fit(self, real_images, batch_size, nb_epoch,
            verbose=1, param_dir="./params", log_csv_path=None, **visualize_params):
        """
        DCGANの訓練を行う関数
        :param real_images: 入力画像データセット
        :param batch_size: バッチサイズ
        :param nb_epoch: 学習回数
        :param verbose: 冗長モード
        :param param_dir: 学習したパラメータのファイルを保存する場所
        :param log_csv_path: 学習結果のcsvファイルを出力するパス
        :param visualize_params: 訓練中に生成をする。そのためのパラメータ。辞書クラス
        :return:

        visualize_params は以下のキーを持つ
        visualize_steps: 何epoch学習した後、生成するか
        dst_dir: 生成した画像を保存するディレクトリパス
        nb_generate: 一度に何枚生成するか
        """

        self.on_training = True
        visualize_params.setdefault('visualize_steps', 1)
        visualize_params.setdefault('dst_dir', "./generated")
        visualize_params.setdefault('nb_generate', 20)

        visualize_steps = visualize_params['visualize_steps']
        visualize_dir = visualize_params['dst_dir']
        nb_generate = visualize_params['nb_generate']

        # Create Directories
        if not os.path.exists(visualize_dir):
            os.makedirs(visualize_dir, exist_ok=True)

        if not os.path.exists(param_dir):
            os.makedirs(param_dir, exist_ok=True)

        # Open csv
        if log_csv_path:
            f = open(log_csv_path, 'w')
            writer = csv.writer(f, lineterminator='\n')

        # Train
        for epoch in range(nb_epoch):
            print("\nepoch : {0}".format(epoch))

            # Shuffle Images
            np.random.shuffle(real_images)
            # Create Noises
            noises = self.make_noises(size=(len(real_images), self.input_dim))

            steps_per_epoch = len(real_images) // batch_size if len(real_images) % batch_size == 0 \
                else len(real_images) // batch_size + 1

            for iter_ in range(steps_per_epoch):
                if verbose == 1:
                    sys.stdout.write("\riter : %d / " % (iter_ * batch_size))
                    sys.stdout.write("%d" % len(real_images))
                    sys.stdout.flush()

                # Create MiniBatch
                real_batch = real_images[iter_ * batch_size: (iter_ + 1) * batch_size]
                noise_batch = noises[iter_ * batch_size: (iter_ + 1) * batch_size]

                # Update Discriminator
                generated_images = self.generator.predict(noise_batch)
                x = np.append(generated_images, real_batch, axis=0)
                y = np.array([0] * len(generated_images) + [1] * len(real_batch))
                x, y = self.shuffle(x, y)
                loss_d, acc_d = self.discriminator.train_on_batch(x, y)

                # Update Generator
                y = np.array([1] * len(generated_images))
                loss_g, acc_g = self.dcgan.train_on_batch(noise_batch, y)

                if verbose == 1:
                    sys.stdout.write(" loss_g : %f" % loss_g)
                    sys.stdout.write(" acc_g : %f" % acc_g)
                    sys.stdout.write(" loss_d : %f" % loss_d)
                    sys.stdout.write(" acc_d : %f" % acc_d)
                    sys.stdout.flush()

                # Write Datas to csv
                if log_csv_path:
                    writer.writerow([epoch, iter_, loss_g, acc_g, loss_d, acc_d])

            # Generate Images from Noises
            if epoch % visualize_steps == 0:
                dst_dir = os.path.join(visualize_dir, "epoch_{}".format(epoch))
                if not os.path.exists(dst_dir):
                    os.mkdir(dst_dir)
                self.generate(dst_dir=dst_dir, nb_generate=nb_generate)

            # Save Parameters
            param_discriminator = os.path.join(param_dir, "discriminator_{}.hdf5".format(epoch))
            param_generator = os.path.join(param_dir, "generator_{}.hdf5".format(epoch))
            self.discriminator.save_weights(param_discriminator)
            self.generator.save_weights(param_generator)

        if log_csv_path:
            f.close()
        self.on_training = False

    def generate(self, dst_dir, nb_generate):
        """
        一様乱数を生成し、その乱数から画像を生成する関数
        :param dst_dir: 画像の保存先ディレクトリ
        :param nb_generate: 何枚生成するか
        :return:
        """
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)

        if not self.on_training:
            if os.path.exists(self.param_path_generator):
                self.generator.load_weights(self.param_path_generator)
            else:
                print("Could not find " + self.param_path_generator)
                exit()

        noises = self.make_noises(size=(nb_generate, self.input_dim))
        generated_images = self.generator.predict(noises)

        for index, img in enumerate(generated_images):
            dst_path = os.path.join(dst_dir,  str(index) + ".png")
            self.save_image(dst_path, img)

    @staticmethod
    def shuffle(x, y):
        assert len(x) == len(y)

        indexes = np.arange(len(x))
        indexes = np.random.permutation(indexes)
        return x[indexes], y[indexes]

    @staticmethod
    def save_image(dst_path, src_img):
        if K.image_dim_ordering() == 'th':
            dst_img = src_img.transpose(1, 2, 0)
        else:
            dst_img = src_img

        if dst_img.shape[2] == 1:
            dst_img = dst_img.reshape(dst_img.shape[:2])

        dst_img = ((dst_img / 2 + 0.5) * 255).astype('uint8')
        cv2.imwrite(dst_path, dst_img)

    @staticmethod
    def make_noises(size):
        random_state = np.random.RandomState(18)
        noises = random_state.uniform(-1., 1., size=size)
        return noises
