import keras.backend as K
from keras.layers import (Lambda, Dense, Reshape, Concatenate)


def MinibatchDisrimination(num_kernels=100, dim_per_kernel=5):
    def f(x):
        _x = Dense(num_kernels * dim_per_kernel, use_bias=False, activation=None)(x)
        _x = Reshape((num_kernels, dim_per_kernel))(_x)

        def mbd(x):
            diffs = K.expand_dims(x, 3) - K.expand_dims(K.permute_dimensions(x, [1, 2, 0]), 0)
            abs_diffs = K.sum(K.abs(diffs), 2)
            x = K.sum(K.exp(-abs_diffs), 2)
            return x

        def lambda_output(input_shape):
            return input_shape[:2]

        return Lambda(mbd, output_shape=lambda_output)
    return f