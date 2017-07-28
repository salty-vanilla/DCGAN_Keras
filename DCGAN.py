from keras.layers import Input
from keras.models import Model
from keras import backend as K
import numpy as np
import sys
import csv
import os
import cv2


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
        self.param_path_generator = param_path_generator
        self.param_path_discriminator = param_path_discriminator
        self.save_steps = None

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
            verbose=1, param_dir="./params", log_csv_path=None,
            save_steps=5, is_shuffle=True,
            is_separate=False, is_noisy_label=False, **visualize_params):
        """
        DCGANの訓練を行う関数
        :param real_images: 入力画像データセット
        :param batch_size: バッチサイズ
        :param nb_epoch: 学習回数
        :param verbose: 冗長モード
        :param param_dir: 学習したパラメータのファイルを保存する場所
        :param log_csv_path: 学習結果のcsvファイルを出力するパス
        :param save_steps: 何epochごとに保存するか
        :param is_shuffle: 入力データをシャッフルするか
        :param is_separate: realとnoisyを別々に学習するか
        :param is_noisy_label: labelにnoiseを付与するか
        :param visualize_params: 訓練中に生成をする。そのためのパラメータ。辞書クラス
        :return:

        visualize_params は以下のキーを持つ
        visualize_steps: 何epoch学習した後、生成するか
        nb_generate: 一度に何枚生成するか
        """

        self.save_steps = save_steps
        visualize_params.setdefault('visualize_steps', 1)
        visualize_params.setdefault('dst_dir', "./generated")
        visualize_params.setdefault('nb_generate', 20)
        visualize_params.setdefault('seed', 18)

        visualize_steps = visualize_params['visualize_steps']
        nb_generate = visualize_params['nb_generate']
        visualize_seed = visualize_params['seed']

        # Create Directories
        if not os.path.exists(param_dir):
            os.makedirs(param_dir, exist_ok=True)

        # Open csv
        if log_csv_path:
            result_dir = os.path.dirname(log_csv_path)
            os.makedirs(result_dir, exist_ok=True)
            f = open(log_csv_path, 'w')
            writer = csv.writer(f, lineterminator='\n')

        # Train
        for epoch in range(1, nb_epoch + 1):
            print("\nepoch : {0}".format(epoch))

            # Shuffle Images
            if is_shuffle:
                np.random.shuffle(real_images)
            # Create Noises
            noises = self.make_noises(size=(len(real_images), self.input_dim))

            steps_per_epoch = len(real_images) // batch_size if len(real_images) % batch_size == 0 \
                else len(real_images) // batch_size + 1

            for iter_ in range(steps_per_epoch):
                # Create MiniBatch
                real_batch = real_images[iter_ * batch_size: (iter_ + 1) * batch_size]
                noise_batch = noises[iter_ * batch_size: (iter_ + 1) * batch_size]

                # Update
                loss_g, acc_g, loss_d, acc_d = \
                    self.update(real_batch, noise_batch, is_separate, is_noisy_label)

                if verbose == 1:
                    self.display(iter_, batch_size, len(real_images), loss_g, acc_g, loss_d, acc_d)

                # Write Datas to csv
                if log_csv_path:
                    writer.writerow([epoch, iter_, loss_g, acc_g, loss_d, acc_d])

            # Generate Images from Noises
            if epoch % visualize_steps == 0 or epoch == 1:
                visualize_dir = os.path.join(result_dir, "epoch_{}".format(epoch))
                self.generate(dst_dir=visualize_dir, nb_generate=nb_generate, seed=visualize_seed)

            # Save Parameters
            self.save_params(param_dir, epoch)

        if log_csv_path:
            f.close()

    def generate(self, dst_dir, nb_generate, seed=None):
        """
        一様乱数を生成し、その乱数から画像を生成する関数
        :param dst_dir: 画像の保存先ディレクトリ
        :param nb_generate: 何枚生成するか
        :param seed: 乱数のシード
        :return:
        """
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir, exist_ok=True)

        noises = self.make_noises(size=(nb_generate, self.input_dim), seed=seed)
        generated_images = self.generator.predict(noises)

        for index, img in enumerate(generated_images):
            dst_path = os.path.join(dst_dir,  str(index) + ".png")
            self.save_image(dst_path, img)

    def update(self, real_batch, noise_batch, is_separate, is_noisy_label):
        loss_d, acc_d = self.update_discriminator(real_batch, noise_batch, is_separate, is_noisy_label)
        loss_g, acc_g = self.update_generator(noise_batch, is_noisy_label)
        return loss_g, acc_g, loss_d, acc_d

    def update_discriminator(self, real_batch, noise_batch, is_separate, is_noisy_label):
        if is_separate:
            # Update Discriminator
            generated = self.generator.predict(noise_batch)
            y = np.array([0] * len(generated))
            if is_noisy_label:
                y = self.make_noisy_label(y)
            loss_d_f, acc_d_f = self.discriminator.train_on_batch(generated, y)
            y = np.array([1] * len(real_batch))
            if is_noisy_label:
                y = self.make_noisy_label(y)
            loss_d_r, acc_d_r = self.discriminator.train_on_batch(real_batch, y)
            loss_d = (loss_d_f + loss_d_r) / 2
            acc_d = (acc_d_f + acc_d_r) / 2
        else:
            # Update Discriminator
            generated = self.generator.predict(noise_batch)
            x = np.append(generated, real_batch, axis=0)
            y = np.array([0] * len(generated) + [1] * len(real_batch))
            if is_noisy_label:
                y = self.make_noisy_label(y)
            loss_d, acc_d = self.discriminator.train_on_batch(x, y)
        return loss_d, acc_d

    def update_generator(self, noise_batch, is_noisy_label):
        y = np.array([1] * len(noise_batch))
        if is_noisy_label:
            y = self.make_noisy_label(y)
        loss_g, acc_g = self.dcgan.train_on_batch(noise_batch, y)
        return loss_g, acc_g

    def save_params(self, param_dir, epoch):
        # Save Parameters
        if epoch % self.save_steps == 0:
            param_discriminator = os.path.join(param_dir, "discriminator_{}.hdf5".format(epoch))
            param_generator = os.path.join(param_dir, "generator_{}.hdf5".format(epoch))
            self.discriminator.save_weights(param_discriminator)
            self.generator.save_weights(param_generator)

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
    def make_noises(size, seed=None):
        if seed is not None:
            random_state = np.random.RandomState(seed)
            noises = random_state.uniform(-1., 1., size=size)
        else:
            noises = np.random.uniform(-1., 1., size=size)
        return noises

    @staticmethod
    def make_noisy_label(label):
        _label = label.astype('float32')
        noise = (np.random.rand(len(label)) * 0.6) - 0.3
        _label += noise
        for i in range(len(_label)):
            if _label[i] < 0:
                _label[i] = 0
        return _label

    @staticmethod
    def display(iter_, batch_size, num_data, loss_g, acc_g, loss_d, acc_d):
        sys.stdout.write("\riter : %d / " % (iter_ * batch_size))
        sys.stdout.write("%d" % num_data)
        sys.stdout.write(" loss_g : %f" % loss_g)
        sys.stdout.write(" acc_g : %f" % acc_g)
        sys.stdout.write(" loss_d : %f" % loss_d)
        sys.stdout.write(" acc_d : %f" % acc_d)
        sys.stdout.flush()
