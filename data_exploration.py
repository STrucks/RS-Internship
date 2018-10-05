from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np


def show_GT():
    f = loadmat("data/indian_pines_gt.mat")
    GT = f['indian_pines_gt']

    plt.imshow(GT)
    plt.show()
    f.clear()


def show_raw_data(index = [0,50,100,150,200]):
    f = loadmat("data/indian_pines.mat")
    data = f['indian_pines']

    for i, ind in enumerate(index):
        plt.subplot(len(index), 1, i+1)
        plt.imshow(data[:, :, ind])
    plt.show()


def show_corrected_data(index = [0,50,100,150,200]):
    f = loadmat("data/indian_pines_corrected.mat")
    data = f['indian_pines_corrected']

    for i, ind in enumerate(index):
        plt.subplot(len(index), 1, i+1)
        plt.imshow(data[:, :, ind])
    plt.show()


def show_combined(index = [0,50,100,150, 199]):

    # comparison gets weird after index 104, because they removed [104-108], [150-163] and 220.

    fig, axes = plt.subplots(nrows=len(index), ncols=3)

    f = loadmat("data/indian_pines.mat")
    data = f['indian_pines']

    for i, ind in enumerate(index):
        axes[i,0].imshow(data[:, :, ind])

    f = loadmat("data/indian_pines_corrected.mat")
    data = f['indian_pines_corrected']

    for i, ind in enumerate(index):
        axes[i, 1].imshow(data[:, :, ind])

    f = loadmat("data/indian_pines_gt.mat")
    GT = f['indian_pines_gt']

    for i in range(len(index)):
        axes[i, 2].imshow(GT)

    plt.show()


def show_AwA():
    f = loadmat("data/AwA_data/attribute/Z_s_con.mat")
    # It consists of 30475 images of 50 animals classes with six pre-extracted feature representations for each image
    print(f.keys())
    print(f['features'].shape)
    print(f['labels'].shape)
    print(f['image_files'].shape)
    print(f['image_files'][0])

    GT = np.reshape(f['features'][:, 0], newshape=(32, 64))
    print(GT)
    plt.imshow(GT)
    plt.show()
    f.clear()


def show_distribution_IP():
    f = loadmat("data/indian_pines.mat")
    raw_data = f['indian_pines']

    f = loadmat("data/indian_pines_gt.mat")
    GT = f['indian_pines_gt']

    data = {}

    for row in range(len(GT)):
        for col in range(len(GT[row, :])):
            if str(GT[row, col]) in data:
                data[str(GT[row, col])].append(raw_data[row, col, :])
            else:
                data[str(GT[row, col])] = []
                data[str(GT[row, col])].append(raw_data[row, col, :])

    data_distr = [(x, len(data[x])) for x in sorted(data.keys())]
    plt.bar([int(x) for x, y in data_distr[1:]], [y for x, y in data_distr[1:]])
    plt.xlabel("label")
    plt.ylabel("amount")
    plt.show()
    """
    DATA DISTRIBUTION:
    print([(x, len(data[x])) for x in sorted(data.keys())])
    [('0', 10776), ('1', 46), ('10', 972), ('11', 2455), ('12', 593), ('13', 205), ('14', 1265), ('15', 386), ('16', 93), ('2', 1428), ('3', 830), ('4', 237), ('5', 483), ('6', 730), ('7', 28), ('8', 478), ('9', 20)]
    """


show_distribution_IP()
