from keras.layers import (
    Input,
    Conv2D,
    LeakyReLU,
    Flatten,
    Dense,
    Activation,
    BatchNormalization,
    Lambda,
    Reshape,
    Concatenate
)
import keras.backend as K
from keras.models import Model
from layers import MinibatchDisrimination as mbd

def get_discriminator(input_shape, is_plot=False):
    print("Building Discriminator ...   ", end="")
    inputs = Input(shape=input_shape)

    # (160, 128, 3) to (80, 64, 32)
    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same')(inputs)
    x = LeakyReLU(0.2)(x)

    # (80, 64, 32) to (40, 32, 64)
    x = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)

    # (40, 32, 64) to (20, 16, 128)
    x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)

    # (20, 16, 128) to (10, 8, 256)
    x = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)

    # (10, 8, 128) to (5, 4, 512)
    x = Conv2D(512, (3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)

    x = Flatten()(x)

    num_kernels = 100
    dim_per_kernel = 5
    M = Dense(num_kernels * dim_per_kernel, use_bias=False, activation=None)
    # M.trainable = False
    MBD = Lambda(minb_disc, output_shape=lambda_output)
    x_mbd = M(x)
    x_mbd = Reshape((num_kernels, dim_per_kernel))(x_mbd)
    x_mbd = MBD(x_mbd)
    x = Concatenate()([x, x_mbd])

    # Input dim : (20480, )
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, x, name='discriminator')

    if is_plot:
        from keras.utils.vis_utils import plot_model
        plot_model(model, to_file='model_discriminator.png', show_shapes=True)

    print("COMPLETE")
    return model


def minb_disc(x):
    diffs = K.expand_dims(x, 3) - K.expand_dims(K.permute_dimensions(x, [1, 2, 0]), 0)
    abs_diffs = K.sum(K.abs(diffs), 2)
    x = K.sum(K.exp(-abs_diffs), 2)
    return x


def lambda_output(input_shape):
    return input_shape[:2]