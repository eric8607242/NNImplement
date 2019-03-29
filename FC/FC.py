import os
import numpy as np
import pickle

class FC:
    def __init__(self, hidden_size, reg, learning_rate, std):
        """
        Args:
            hidden_size(list):the list contained the each size of the hidden layer
            reg(float):the regularization strength
            learning_rate(float):learning rate in updating step
            std(float): the scale in the weight initialization

        Attriibutes:
            hidden_size(list):the list contained the each size of the hidden layer
            reg(float):the regularization strength
            learning_rate(float): learning rate in updating step
            std(float): the scale in the weignt initialization

            W(dict): a dict contained the weight of each layer
            b(dict): a dict contained the bias of the each layer
        """
        self.hidden_size = hidden_size
        self.reg = reg
        self.learning_rate = learning_rate
        self.std = std

        self.data_size = 32*32*3
        self.train_size = 49000
        self.val_size = 1000
        self.test_size = 1000
        self.class_num = 10

        self.train_X, self.train_Y = None, None
        self.val_X, self.val_Y = None, None
        self.test_X, test_Y = None, None

        self.W, self.b = {}, {}
        self.output_W, self.output_b = None, None

    def _init_weight(self):
        pre_layer_size = self.data_size
        for l in range(len(self.hidden_size)):
            self.W[l] = np.random.normal(0, self.std, (pre_layer_size, self.hidden_size[l]))
            self.b[l] = np.zeros(self.hidden_size[l])

        self.output_W = np.random.normal(0, self.std, (self.hidden_size[-1], self.class_num))
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
        self.test_X, self.test_y = self._load_cifar(filename)

        self.train_X = self.train_X.reshape(self.train_size, -1)
        self.val_X = self.val_X.reshape(self.val_size, -1)
        self.test_X = self.test_X.reshape(self.test_size, -1)
        
    def _forward(self, X):
        cache= {}
        pre_layer = X

        for l in range(len(self.hidden_size)):
            cache[]
           

    def _backward(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass

if __name__ == '__main__':
    fc = FC([10,10,10,10], 1e-4, 1e-4, 1e-1)
    fc.load_data()
