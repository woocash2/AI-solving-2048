import numpy as np


def tanh(X):
    return np.tanh(X)


def dtanh(X):
    return 1. - np.multiply(tanh(X), tanh(X))


def relu(X):
    return X * (X > 0.)


def drelu(X):
    return 1. * (X > 0.)


def sigmoid(X):
    return 1. / (1. + np.exp(-X))


def dsigmoid(X):
    return np.multiply(sigmoid(X), (1. - sigmoid(X)))


activation = {'relu': relu, 'sigmoid': sigmoid, 'tanh': tanh}
dactivation = {'relu': drelu, 'sigmoid': dsigmoid, 'tanh': dtanh}


class Dense:

    def __init__(self, input_size, output_size, activation=None):
        self.insize = input_size
        self.outsize = output_size
        self.activ = activation

        self.W = np.random.uniform(-np.sqrt(6. / (input_size + output_size)), np.sqrt(6. / (input_size + output_size)), (input_size, output_size))
        self.B = np.random.uniform(-np.sqrt(6. / (output_size)), np.sqrt(6. / (output_size)), output_size)

        self.X = None
        self.Y = None
        self.Z = None

    def affine(self, X):
        Y = np.matmul(X, self.W)
        Y += self.B
        return Y

    def forward(self, X):
        self.X = X
        self. Y = self.affine(X)
        self.Z = self.Y if self.activ not in activation else activation[self.activ](self.Y)
        return self.Z

    def update_params_and_chain(self, D, lr):
        A = self.Y if self.activ not in dactivation else dactivation[self.activ](self.Y)
        E = np.multiply(A, D)
        gB = np.mean(E, axis=0)
        gW = np.matmul(np.transpose(self.X), E)
        F = np.matmul(E, np.transpose(self.W))

        self.B -= lr * gB
        self.W -= lr * gW

        return F


class Convolutional:

    def __init__(self, height, width, input_channels, output_channels, kernel_size, activation=None):
        self.height = height
        self.width = width
        self.channels = input_channels
        self.knum = output_channels
        self.ksize = kernel_size
        self.activ = activation

        self.kernels = []

        for i in range(output_channels):
            K = np.random.uniform(-np.sqrt(6. / (2. * kernel_size * kernel_size)), np.sqrt(6. / (2. * kernel_size * kernel_size)), size=(kernel_size, kernel_size, input_channels))
            self.kernels.append(K)

        self.bias = np.random.uniform(-np.sqrt(6. / (2. * (width - kernel_size + 1) * (height - kernel_size + 1))), np.sqrt(6. / (2. * (width - kernel_size + 1) * (height - kernel_size + 1))), size=(height - kernel_size + 1, width - kernel_size + 1, output_channels))

        self.X = None
        self.Y = None
        self.Z = None

    def conv_at(self, X, kernel, xsrc, ysrc):
        A = np.multiply(kernel, X[:, xsrc:xsrc + self.ksize, ysrc:ysrc + self.ksize, :])
        return np.sum(A, axis=(1, 2, 3))

    def forward(self, X):
        self.X = X
        self.Y = np.zeros((len(self.X), self.height - self.ksize + 1, self.width - self.ksize + 1, self.knum))

        for j, kernel in enumerate(self.kernels):
            for xsrc in range(self.height - self.ksize + 1):
                for ysrc in range(self.width - self.ksize + 1):
                    self.Y[:, xsrc, ysrc, j] = self.conv_at(X, kernel, xsrc, ysrc)

        self.Y += self.bias
        self.Z = self.Y if self.activ not in activation else activation[self.activ](self.Y)
        return self.Z

    def update_params_and_chain(self, D, lr):
        A = self.Y if self.activ not in dactivation else dactivation[self.activ](self.Y)
        D = np.multiply(A, D)
        gB = np.mean(D, axis=0)

        gK = np.zeros((self.knum, self.ksize, self.ksize, self.channels))
        E = np.zeros(self.X.shape)

        for k, kernel in enumerate(self.kernels):
            kernel_tensor = np.array([kernel] * len(D))
            for xsrc in range(self.height - self.ksize + 1):
                for ysrc in range(self.width - self.ksize + 1):
                    K = np.multiply(D[:, xsrc, ysrc, k], np.transpose(self.X[:, xsrc:xsrc + self.ksize, ysrc:ysrc + self.ksize, :]) / float(len(self.X)))
                    gK[k] += np.sum(np.transpose(K), axis=0)
                    M = np.multiply(D[:, xsrc, ysrc, k], np.transpose(kernel_tensor))
                    E[:, xsrc:xsrc + self.ksize, ysrc:ysrc + self.ksize, :] += np.transpose(M)

        self.kernels -= lr * gK
        self.bias -= lr * gB
        return E


class Flatten:

    def __init__(self, input_height, input_width, input_channels):
        self.h = input_height
        self.w = input_width
        self.c = input_channels

    def forward(self, X):
        Y = np.zeros((len(X), self.h * self.w * self.c))
        for i, x in enumerate(X):
            Y[i] = x.flatten()
        return Y

    def update_params_and_chain(self, D, lr):
        E = np.zeros((len(D), self.h, self.w, self.c))
        for i, d in enumerate(D):
            E[i] = d.reshape(self.h, self.w, self.c)
        return E
