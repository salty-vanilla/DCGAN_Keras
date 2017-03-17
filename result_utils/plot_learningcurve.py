import csv
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def plot(src_path, loss_curve_path, acc_curve_path):
    f = open(src_path, 'r')
    reader = csv.reader(f)

    datas = np.array([row for row in reader])
    loss_g, acc_g, loss_d, acc_d = zip(*datas[:, 2:])

    plt.figure(1)
    plt.plot(loss_g, label="loss_g", lw=3)
    plt.plot(loss_d, label="loss_d", lw=3)
    plt.legend(loc=0, fontsize=24)
    plt.savefig(loss_curve_path)

    plt.figure(2)
    plt.plot(acc_g, label="acc_g", lw=3)
    plt.plot(acc_d, label="acc_d", lw=3)
    plt.legend(loc=0, fontsize=24)
    plt.savefig(acc_curve_path)


def main():
    src_path = "../result.csv"
    plot(src_path, loss_curve_path="loss.png", acc_curve_path="acc.png")


if __name__ == "__main__":
    main()
