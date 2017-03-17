import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import cv2


def plot_result(dst_path, images, shape, labels):
    plt.figure()

    for j in range(shape[0]):
        plt.subplot(shape[0], shape[1] + 1, j * (shape[1] + 1) + 1)
        plt.xticks([])
        plt.yticks([])
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.xlim(-1., 1.)
        plt.ylim(-1., 1.)
        plt.text(-1., 0., labels[j])
        for i in range(shape[1]):
            img = images[j * shape[1] + i]
            plt.subplot(shape[0], shape[1] + 1, j * (shape[1] + 1) + i + 2)
            plt.xticks([])
            plt.yticks([])
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            plt.imshow(img, cmap='gray')

    plt.savefig(dst_path)


def main():
    nb_per_epoch = 5
    dst_path = "result.png"
    shape = (4, 5)
    labels = ["epoch_0", "epoch_10", "epoch_50", "epoch_100"]
    src_dirs = ["../generated/epoch_0/",
                "../generated/epoch_10/",
                "../generated/epoch_50/",
                "../generated/epoch_100/"]

    image_paths = [os.path.join(src_dir, file) for src_dir in src_dirs
                   for file in os.listdir(src_dir)[:nb_per_epoch]]

    images = [cv2.imread(path, 0) for path in image_paths]

    plot_result(dst_path, images, shape, labels)


if __name__ == "__main__":
    main()
