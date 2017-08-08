import numpy as np
import cv2
import os
from keras.preprocessing.image import Iterator
from keras import backend as K
from abc import abstractmethod


class DataGenerator:
    def __init__(self, image_dir, target_size=None, color_mode='rgb', nb_samples=None):
        self.image_dir = image_dir
        self.target_size = target_size
        self.color_mode = color_mode
        self.image_paths = np.array(sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)]))
        if nb_samples is not None:
            self.image_paths = self.image_paths[:nb_samples]
        self.data_num = len(self.image_paths)

    @abstractmethod
    def flow(self, batch_size, shuffle=True):
        return DataIterator(paths=self.image_paths, target_size=self.target_size,
                            batch_size=batch_size, shuffle=shuffle)


class DataIterator(Iterator):
    def __init__(self, paths, target_size=None, color_mode='rgb',
                 batch_size=32, shuffle=True, seed=None, is_loop=True):
        self.paths = paths
        self.target_size = target_size
        self.color_mode = color_mode
        self.is_loop = is_loop
        self.data_num = len(self.paths)
        super().__init__(self.data_num, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        image_path_batch = self.paths[index_array]
        image_batch = np.array([load_image(path, self.target_size, self.color_mode)
                                for path in image_path_batch])
        return image_batch


def load_image(path, target_size=None, color_mode='rgb'):
    assert color_mode in ['grayscale', 'gray', 'rgb']
    if color_mode in ['grayscale', 'gray']:
        imread_flag = cv2.IMREAD_GRAYSCALE
        channel = 1
    else:
        imread_flag = cv2.IMREAD_COLOR
        channel = 3

    src = cv2.imread(path, imread_flag)

    if target_size is not None:
        src = cv2.resize(src, target_size, interpolation=cv2.INTER_LINEAR)

    h, w = src.shape[:2]
    if K.image_dim_ordering() == 'th':
        output_shape = (channel, h, w)
    else:
        output_shape = (h, w, channel)
    src = normalize(src)
    return src.reshape(output_shape)


def normalize(x):
    return (x.astype('float32') / 255 - 0.5) / 0.5
