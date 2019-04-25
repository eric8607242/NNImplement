import pickle
import numpy as np

class CNN:
    def __init__(self, reg, learning_rate, std):

        self.hidden_size = [6, 16, 120, 84]
        self.reg = reg
        self.learning_rate = learning_rate
        self.std = std

        self.data_size = 32*32*3
        self.train_size = 49000
        self.val_size = 1000
        self.test_size = 10000
        self.class_num = 10

        self.train_X, self.train_Y = None, None
        self.val_X, self.val_Y = None, None
        self.test_X, self.val_Y = None, None

        self.conv_layer_num = 2
        self.filter_size = 5

        self.W, self.b = {}, {}
        self.output_W, self.output_b = None, None

        self._init_weight()

    def _init_weight(self):

        filter_depth = 3
        for l in range(self.conv_layer_num):
            self.W[l] = np.random.normal(0.0, self.std, (self.hidden_size[l], self.filter_size, self.filter_size, filter_depth))
            self.b[l] = np.zeros(self.hidden_size[l])
            filter_depth = self.hidden_size[l]

        layer_dim = self.filter_size*self.filter_size*self.hidden_size[self.conv_layer_num-1]
        for l in range(self.conv_layer_num, len(self.hidden_size)):
            self.W[l] = np.random.normal(0.0, self.std, (layer_dim, self.hidden_size[l]))
            self.b[l] = np.zeros(self.hidden_size[l])

            layer_dim = self.hidden_size[l]

        self.output_W = np.random.normal(0.0, self.std, (layer_dim, self.class_num))
        self.output_b = np.zeros(self.class_num)

    def _load_cifar(self, filename):
        with open(filename, 'rb') as f:
            datadict = pickle.load(f, encoding='latin1')
            X = datadict['data']
            Y = datadict['labels']
            X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
            Y = np.array(Y)
            return X, Y

    def load_data(self):
        xs = []
        ys = []

        for b in range(1, 6):
            filename = './cifar-10-batches-py/data_batch_' + str(b)

            X, Y = self._load_cifar(filename)
            xs.append(X)
            ys.append(Y)

        self.train_X = np.concatenate(xs)
        self.train_Y = np.concatenate(ys)

        self.val_X = self.train_X[-1*self.val_size:]
        self.val_Y = self.train_Y[-1*self.val_size:]

        self.train_X = self.train_X[:-1*self.val_size]
        self.train_Y = self.train_Y[:-1*self.val_size]

        filename = './cifar-10-batches-py/test_batch'
        self.test_X, self.test_Y = self._load_cifar(filename)


    def _forward_pooling(self):
        pass

    def _backward_pooling(self):
        pass

    def _forward_conv(self):
        pass

    def _backward_conv(self):
        pass

    def train(self):
        pass

if __name__ == '__main__':
    cnn = CNN(0.25, 0.24, 1e-2)
    cnn.load_data()
