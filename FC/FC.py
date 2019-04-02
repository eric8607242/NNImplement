import os
import matplotlib.pyplot as plt
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
            layers_size(int): the size of the hidden layers
            reg(float):the regularization strength
            learning_rate(float): learning rate in updating step
            std(float): the scale in the weignt initialization

            data_size(int): the size of a training data
            train_size(int): the numbers of the training data
            val_size(int): the numbers of the validation data
            test_size(int): the numbers of the testing data
            class_num(int): the number of the classes in data set

            train_X(ndarray): training data
            train_Y(ndarray): training labels
            val_X(ndarray): validation data
            val_Y(ndarray): validation labels
            test_X(ndarray): testing data
            test_Y(ndarray): testing labels

            W(dict): a dict contained the weight of each layer
            b(dict): a dict contained the bias of the each layer
            output_W(ndarray): the weight in output layer
            output_b(ndarray): the bias in output layer
        """
        self.hidden_size = hidden_size
        self.layers_size = len(self.hidden_size)
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
        self.test_X, test_Y = None, None

        self.W, self.b = {}, {}
        self.output_W, self.output_b = None, None

    def _init_weight(self):
        """Initialize the weight and bias accroding the size of the each layer user defined
        Use the normal distribution which center is 0 and the sacle size is user defined"""
        pre_layer_size = self.data_size
        for l in range(self.layers_size):
            self.W[l] = np.random.normal(0, self.std, (pre_layer_size, self.hidden_size[l]))
            self.b[l] = np.zeros(self.hidden_size[l])
            pre_layer_size = self.hidden_size[l]

        self.output_W = np.random.normal(0, self.std, (self.hidden_size[-1], self.class_num))
        self.output_b = np.zeros(self.class_num)

    def _load_cifar(self, filename):
        """Load the each file of cifar and reshape the data"""
        with open(filename, 'rb') as f:
            datadict = pickle.load(f, encoding='latin1')
            X = datadict['data']
            Y = datadict['labels']
            X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
            Y = np.array(Y)
            return X, Y
     
    def load_data(self):
        """Load data
        Load the cifar data and partition into val data and train data.Reshape all data to 2 dimension ndarray and normalize by sub the mean of the training data"""
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

        mean_image = self.train_X.mean()
        self.train_X -= mean_image
        self.val_X -= mean_image
        self.test_X -= mean_image

        self.train_X = self.train_X.reshape(self.train_size, -1)
        self.val_X = self.val_X.reshape(self.val_size, -1)
        self.test_X = self.test_X.reshape(self.test_size, -1)

        self._init_weight()
        
    def _forward(self, X):
        """forwarding of the FC
        Args:
            X(ndarray): the batch data

        Return:
            cache(dict): a dict which has the output of the each layer.
        """
        cache= {}
        pre_layer = X

        for l in range(self.layers_size):
            layer = pre_layer.dot(self.W[l]) + self.b[l]
            relu = np.maximum(layer, 0)
            pre_layer = relu

            cache["l_"+str(l)] = layer
            cache["relu_"+str(l)] = relu

        cache["scores"] = pre_layer.dot(self.output_W) + self.output_b

        return cache
           
    def _backward(self, X, Y, cache):
        """backward of the FC
        Args:
            X(ndarray): the batch data
            X(ndarray): the batch labels
            cache(dict): a dict which is generated by _forward().There are the output of each layer.
        Returns:
            d_cache(dict): a dict which has the gradient of the each weight and bias.
        """
        cache["relu_-1"] = X
        N = X.shape[0]
        d_cache = {}

        scores = cache["scores"]
        scores -= np.matrix(np.max(scores, axis=1)).T
        p = np.exp(scores) / np.sum(np.exp(scores), axis=1).reshape(N, 1)
        p[np.arange(N), Y] -= 1
        dloss = p/N
 
        d_cache["o_W"] = np.dot(cache["relu_"+str(self.layers_size-1)].T, dloss) + self.reg*self.output_W
        d_cache["o_b"] = np.sum(dloss, axis=0)

        dscores = np.dot(dloss, self.output_W.T)
        dout = dscores

        for l in reversed(range(self.layers_size)):
            drelu = np.zeros_like(cache["relu_"+str(l)])
            drelu[cache["relu_"+str(l)] > 0] = 1
            drelu = dout * drelu

            d_cache["W_"+str(l)] = np.dot(cache["relu_" + str(l-1)].T, drelu) + self.reg*self.W[l]
            d_cache["b_"+str(l)] = np.sum(drelu, axis=0)

            dlayer = np.dot(drelu, self.W[l].T)
            dout = dlayer
        return d_cache

    def _update_weight(self, learning_rate, d_cache):
        """Use SGD to update the weight accroding the gradient of each weight and bias.
        Args:
            learning_rate(float): the learning rate decayed in each epoch
            d_cache(dict): the dict has gradient of each weight and bias generated by _backward 
        """
        for l in range(self.layers_size):
            self.W[l] += -1*d_cache["W_"+str(l)] * learning_rate
            self.b[l] += -1*d_cache["b_"+str(l)] * learning_rate

        self.output_W += -1*d_cache["o_W"] * learning_rate
        self.output_b += -1*d_cache["o_b"] * learning_rate

    def train(self, batch_size = 200, epoch = 10, print_f = 0):
        """ Training model
        Args:
            print_f(int): the int determine whether print the loss in specific iteration
        """
        iteration = self.train_size//batch_size
        learning_rate = self.learning_rate

        for e in range(epoch):
            for i in range(iteration):
                batch_X = self.train_X[batch_size*i:batch_size*(i+1)]
                batch_Y = self.train_Y[batch_size*i:batch_size*(i+1)]

                cache = self._forward(batch_X)

                scores = cache["scores"]
                scores -= np.matrix(np.max(scores, axis=1)).T
                correct_score = scores[np.arange(batch_size), batch_Y]

                loss = -correct_score + np.log(np.sum(np.exp(scores), axis=1))
                loss = np.sum(loss)
                loss /= batch_size
                for l in range(self.layers_size):
                    loss += 0.5 * self.reg * (np.sum(self.W[l]*self.W[l]))
                loss += 0.5 * self.reg * (np.sum(self.output_W*self.output_W))

                if i%200 == 0 and print_f == 0:
                    print("Epoch %d ----- Iteration %d ----- Loss %f-----" % (e, i, loss))
                    self.predict(1)

                d_cache = self._backward(batch_X, batch_Y, cache)
                self._update_weight(learning_rate, d_cache)
            learning_rate *= 0.95

    def tuning_hyperparameter(self, learning_range, std_range, interval = 5):
        """Tuning the hyperparameter
        Args:
            learning_range(list): a list which length is 2 indicate the range learning rate will be
            std_range(list): a list which length is 2 indicate the range std will be
            interval(int): the number of fragment range will be spilt
        """
        
        learning_slice = np.linspace(learning_range[0], learning_range[1], num=interval)
        std_slice = np.linspace(std_range[0], std_range[1], num=interval)

        best_hp = {}
        best_hp["acc"] = 0.0

        for l in learning_slice:
            for s in std_slice:
                print("----- learning rate %f ----- std %f -----" % (l, s))
                self.learning_rate = l
                self.std = s

                self._init_weight()
                self.train(epoch=2, print_f=1)

                accuracy = self.predict(data_set=2)

                if accuracy > best_hp["acc"]:
                    best_hp["acc"] = accuracy
                    best_hp["l"] = l
                    best_hp["s"] = s

        print("The best ----- learning rate %f ----- std %f -----" % (best_hp["l"], best_hp["s"]))
        self.learning_rate = best_hp["l"]
        self.std = best_hp["s"]
        self._init_weight()


    def predict(self, data_set = 0):
        """Predict the accuracy
        Args:
            data_set(int): the int to indicate the predict data set(test = 0 and > 2, train = 1, val = 2),
        """
        data_type = "Testing data"
        predict_data = self.test_X
        predict_label = self.test_Y
        if data_set == 1:
            predict_data = self.train_X
            predict_label = self.train_Y
            data_type = "Training data"
        elif data_set == 2:
            predict_data = self.val_X
            predict_label = self.val_Y
            data_type = "Validation data"

        cache = self._forward(predict_data)
        scores = cache["scores"]

        scores_i = np.argmax(scores, axis=1)
        accuracy = (scores_i == predict_label).mean()

        print("----- %s Accuracy ----- %f" % (data_type, accuracy))

        return accuracy
    
    def visualize_data(self):
        """Visualize the data in each class"""
        classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        sample_numbers = 10

        for y, cls in enumerate(classes):
            idxs = np.flatnonzero(self.train_Y == y)
            idxs = idxs[:sample_numbers]
            pictures = self.train_X[idxs]

            for i in range(sample_numbers):
                plt_idx = i * sample_numbers + y + 1
                picture = pictures[i].reshape(32, 32, 3)
                picture = (picture-picture.min()) / (picture.max()-picture.min())
                plt.subplot(sample_numbers, self.class_num, plt_idx)
                plt.imshow(picture)
                plt.axis('off')
        plt.show()

    def visualize_weight(self):
        """Visualize the weight"""
        visual_w = self.W[0].T
        N = visual_w.shape[0]
        row = N//5 + 1

        for n in range(N):
            vw = visual_w[n].reshape(32, 32, 3)
            vw = (vw-vw.min()) / (vw.max()-vw.min())
            plt.subplot(row, 5, n+1)
            plt.imshow(vw)
            plt.axis('off')
        plt.show()

if __name__ == '__main__':
    fc = FC([100, 100, 100], 0.0, 2e-3, 1e-2)
    fc.load_data()
    fc.tuning_hyperparameter([1e-6, 1e-2], [1e-3, 1e-1])
    fc.train()
    fc.predict()
