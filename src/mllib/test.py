import pickle

import tensorflow.keras.datasets.mnist as mnist
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from models import Sequential
from layers import Dense, Convolutional, Flatten
from src.utils.movesprep import data_path
import os


def one_hot_label(digit):
    x = np.zeros(10)
    x[digit] = 1
    return x


def fun_with_tests(model, test_data, test_features, tries):
    for i in range(tries):
        index = random.randint(0, len(test_data[0]))
        y = model.predict(np.array([test_features[0][index]]))
        pred = np.argmax(y)
        print('The number is:', pred)
        plt.imshow(test_data[0][index])
        plt.show()


def decide(model, X):
    Y = model.predict(X)
    Z = []
    for y in Y:
        Z.append(np.argmax(y))
    return Z


def accuracy(model, data, labels):
    good = 0
    num = 0
    for X, Y in list(zip(data, labels)):
        Z = decide(model, X)
        for i in range(len(Z)):
            num += 1
            if Z[i] == np.argmax(Y[i]):
                good += 1
    return good / num


def get_rgb(trd):
    tr_data = np.zeros((len(trd), 28, 28, 3), dtype=float)
    for i in range(len(tr_data)):
        tr_data[i, :, :, 0] = trd[i, :, :]
        tr_data[i, :, :, 1] = trd[i, :, :]
        tr_data[i, :, :, 2] = trd[i, :, :]
    return tr_data


def normalize(trd):
    return trd / 255.

'''
def flatten(trd):
    for i in range(len(tr_data)):
        if flatten:
            train_data.append(tr_data[i].flatten() / 255.)
        else:
            train_data.append(tr_data / 255.)
    for i in range(len(te_data)):
        if flatten:
            test_data.append(te_data[i].flatten() / 255.)
        else:
            test_data.append(te_data[i] / 255.)
    pass
'''


def make_batch(trd, yl, perbatch):
    ntrd = np.zeros((int(len(trd)/perbatch), perbatch, 28, 28, 3))
    ny = np.zeros((int(len(trd)/perbatch), perbatch, 10))
    for i in range(0, len(trd), perbatch):
        ntrd[int(i/perbatch)] = np.array(trd[i:i + perbatch])
        ny[int(i/perbatch)] = np.array([one_hot_label(y) for y in yl[i:i + perbatch]])
    return ntrd, ny

def batched_data(perbatch, rgb=False, flatten=False):
    (tr_data, train_labels), (te_data, test_labels) = mnist.load_data()

    tr_data = normalize(tr_data)
    te_data = normalize(te_data)

    if rgb:
        tr_data = get_rgb(tr_data)
        te_data = get_rgb(te_data)

    x_train, y_train = make_batch(tr_data, train_labels, perbatch)
    x_test, y_test = make_batch(te_data, test_labels, perbatch)

    x_train = np.array(x_train, dtype=float)
    y_train = np.array(y_train, dtype=float)
    x_test = np.array(x_test, dtype=float)
    y_test = np.array(y_test, dtype=float)

    return (x_train, y_train), (x_test, y_test)


def test_dense_grayscale():

    (x_train, y_train), (x_test, y_test) = batched_data(50, rgb=False, flatten=True)

    model = Sequential()
    model.add_layer(Dense(28*28, 64))
    model.add_layer(Dense(64, 64))
    model.add_layer(Dense(64, 10, activation='sigmoid'))

    model.fit(x_train, y_train, 10, 0.1)

    print(accuracy(model, x_train, y_train))
    print(accuracy(model, x_test, y_test))


def test_conv_rgb():
    (x_train, y_train), (x_test, y_test) = batched_data(1000, rgb=True, flatten=False)

    #with open(os.path.join(data_path(), 'mnist-conv-larger-net.pickle'), 'rb') as file:
    #    model = pickle.load(file)



    model = Sequential()
    model.add_layer(Convolutional(28, 28, 3, 4, 9, activation='tanh'))
    model.add_layer(Convolutional(20, 20, 4, 2, 9, activation='tanh'))
    model.add_layer(Flatten(12, 12, 2))
    model.add_layer(Dense(288, 128, activation='sigmoid'))
    model.add_layer(Dense(128, 10, activation='sigmoid'))



    model.fit(x_train, y_train, epochs=1, lr=0.1, verbose=True)
    print(accuracy(model, x_train, y_train))
    print(accuracy(model, x_test, y_test))

    with open(os.path.join(data_path(), 'mnist-conv-larger-net.pickle'), 'wb') as file:
        pickle.dump(model, file)


def measure_accuracy():
    with open(os.path.join(data_path(), 'mnist-conv-larger-net.pickle'), 'rb') as file:
        net = pickle.load(file)
    (x_train, y_train), (x_test, y_test) = batched_data(1000, rgb=True, flatten=False)
    print('train', accuracy(net, x_train, y_train))
    print('test', accuracy(net, x_test, y_test))


def test_net():
    with open(os.path.join(data_path(), 'mnist-conv-larger-net.pickle'), 'rb') as file:
        net = pickle.load(file)
    (x_train, y_train), (x_test, y_test) = batched_data(1000, rgb=True, flatten=False)
    fun_with_tests(net, x_test, x_test, 100)


if __name__ == '__main__':
    t = input()
    if t == 'train':
        test_conv_rgb()
    if t == 'test':
        test_net()
    if t == 'acc':
        measure_accuracy()
