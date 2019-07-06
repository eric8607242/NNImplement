import pickle
import numpy as np

class CNN:
    def __init__(self, reg, learning_rate, std, fc_layers=[120, 84], conv_layers=None):

        if conv_layers == None:
            self.conv_layers = [
                        {
                            "name":"conv",
                            "filter_nums":6,
                            "filter_size":5,
                            "padding":0,
                            "stride":1
                        },{
                            "name":"pool",
                            "filter_size":2,
                            "padding":0,
                            "stride":2
                        },{
                            "name":"conv",
                            "filter_nums":16,
                            "filter_size":5,
                            "padding":0,
                            "stride":1
                        },{
                            "name":"pool",
                            "filter_size":2,
                            "padding":0,
                            "stride":2
                        }
                    ]
        else:
            self.conv_layers = conv_layers

        self.fc_layers = fc_layers
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

        self.conv_W, self.conv_b = {}, {}
        self.fc_W, self.fc_b = {}, {}
        self.output_W, self.output_b = None, None

        self._init_weight()

    def _init_weight(self):
        filter_depth = 3
        filter_nums = None
        filter_size = None

        for l in range(len(self.conv_layers)):
            if self.conv_layers[l]["name"] == "conv":
                filter_nums = self.conv_layers[l]["filter_nums"]
                filter_size = self.conv_layers[l]["filter_size"]

                self.conv_W[l] = np.random.normal(0.0, self.std,
                        (filter_nums, filter_size, filter_size, filter_depth))
                self.conv_b[l] = np.zeros(filter_nums)
                filter_depth = filter_nums

        layer_dim = filter_depth*filter_size*filter_size
        for l in range(len(self.fc_layers)):
            self.fc_W[l] = np.random.normal(0.0, self.std, (layer_dim, self.fc_layers[l]))
            self.fc_b[l] = np.zeros(self.fc_layers[l])

            layer_dim = self.fc_layers[l]

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


    def _pooling_forward(self, X, param):
        stride, padding, pooling_size = param

        N, H, W, C = X.shape

        out_H = 1+int(H-pooling_size)//stride
        out_W = 1+int(W-pooling_size)//stride

        out = np.zeros([N, out_H, out_W, C])

        for n in range(N):
            for n_h in range(out_H):
                for n_w in range(out_W):
                    for c in range(C):
                        region = X[n, n_h*stride:n_h*stride+pooling_size, n_w*stride:n_w*stride+pooling_size, c]
                        out[n, n_h, n_w, c] = np.amax(region)

        cache = (X, param)
        return out, cache


    def _pooling_backward(self, dout, cache):
        X, param = cache
        stride, padding, pooling_size = param
        N, H, W, C = dout.shape

        out_H = 1+int(H-pooling_size)//stride
        out_W = 1+int(W-pooling_size)//stride

        dX = np.zeros_like(X)

        for n in range(N):
            for n_h in range(out_H):
                for n_w in range(out_W):
                    for c in range(C):
                        region = X[n, n_h*stride:n_h*stride+pooling_size, n_w*stride:n_w*stride+pooling_size, c]
                        max_index = np.argmax(region)
                        max_index = np.unravel_index(max_index, (pooling_size, pooling_size))

                        dX[n, n_h*stride+max_index[0], n_w*stride+max_index[1], c] = dout[n, n_h, n_w, c]
        return dX


    def _conv_forward(self, X, W, b, param):
        stride, padding = param

        N, x_H, x_W, C = X.shape
        F, f_H, f_W, C = W.shape

        out_H = int(1 + (x_H+2*padding-f_H)/stride)
        out_W = int(1 + (x_W+2*padding-f_W)/stride)

        out = np.zeros((N, out_H, out_W, F))
        padding_X = np.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)), 'constant')

        for n in range(N):
            for f in range(F):
                for n_h in range(out_H):
                    for n_w in range(out_W):
                        region = padding_X[n, n_h*stride:n_h*stride+f_H, n_w*stride:n_w*stride+f_W, :]
                        out[n][n_h][n_w][f] = np.sum(region*W[f] + b[f])

        cache = (X, W, b, param)
        return out, cache

    def _conv_backward(self, dout, cache):
        X, W, b, param = cache
        stride, padding = param

        N, x_H, x_W, C = X.shape
        F, f_H, f_W, C = W.shape
        N, out_H, out_W, F = dout.shape

        dX = np.zeros_like(X)
        dW = np.zeros_like(W)
        db = np.zeros_like(b)

        for f in range(F):
            db[f] = np.sum(dout[:, :, :, f])

        padding_X = np.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)), 'constant')
        dpadding_X = np.zeros_like(padding_X)


        for n in range(N):
            for f in range(F):
                for n_h in range(out_H):
                    for n_w in range(out_W):
                        region = padding_X[n, n_h*stride:n_h*stride+f_H, n_w*stride:n_w*stride+f_W, :]
                        dW[f] += region * dout[n, n_h, n_w, f]
                        dpadding_X[n, n_h*stride:n_h*stride+f_H, n_w*stride:n_w*stride+f_W, :] += dout[n, n_h, n_w, f] * W[f]

        dX = dpadding_X[:, padding:-padding, padding:-padding, :]

        return dX, dW, db


    def _backward(self, X, Y, cache):
        cache["relu_-1"] = X
        N = X.shape[0]
        d_cache = {}

        scores = cache["scores"]
        scores -= np.matrix(np.max(scores, axis=1)).T
        p = np.exp(scores) / np.sum(np.exp(scores), axis=1).reshape(N, 1)
        p[np.arange(N), Y] -= 1
        dloss = p/N

        d_cache["output_W"] = np.dot(cache["relu_"+str(len(self.fc_layers)-1)].T , dloss) + self.reg*self.output_W
        d_cache["output_b"] = np.sum(dloss, axis=0)

        dscores = np.dot(dloss, self.output_W.T)
        dout = dscores

        for l in reversed(range(len(self.fc_layers))):
            drelu = np.zeros_like(cache["relu_"+str(l)])
            drelu[cache["relu_"+str(l)] > 0] = 1
            drelu = dout * drelu

            d_cache["W_"+str(l)] = np.dot(cache["relu_" + str(l-1)].T, drelu) + self.reg*self.fc_W[l]
            d_cache["b_"+str(l)] = np.sum(drelu, axis=0)

            dlayer = np.dot(drelu, self.fc_W[l].T)
            dout = dlayer
        return dout, d_cache

    def _forward(self, X):
        cache = {}
        pre_layer = X

        for l in range(len(self.fc_layers)):
            layer = pre_layer.dot(self.fc_W[l])+self.fc_b[l]
            relu = np.maximum(layer, 0)
            pre_layer = relu

            cache["l_"+str(l)] = layer
            cache["relu_"+str(l)] = relu

        cache["scores"] = pre_layer.dot(self.output_W) + self.output_b
        return cache


    def _update_weight(self, learning_rate, d_cache, d_conv_cache):
        pass

    def train(self, batch_size = 200, epoch = 10):
        iteration = self.train_size//batch_size

        for e in range(epoch):
            for i in range(iteration):
                batch_X = self.train_X[batch_size*i:batch_size*(i+1)]
                batch_Y = self.train_Y[batch_size*i:batch_size*(i+1)]

                out = batch_X
                conv_cache = {}
                for l in range(len(self.conv_layers)):
                    layer = self.conv_layers[l]
                    stride = layer["stride"]
                    padding = layer["padding"]

                    if layer["name"] == "conv":
                        param = (stride, padding)
                        W, b = self.conv_W[l], self.conv_b[l]
                        out, conv_cache[l] = self._conv_forward(out, W, b, param)
                    elif layer["name"] == "pool":
                        pooling_size = layer["filter_size"]
                        param = (stride, padding, pooling_size)
                        out, conv_cache[l] = self._pooling_forward(out, param)


                flatten_shape = out.shape
                out = out.reshape(batch_size, -1)

                fc_cache = self._forward(out)

                scores = fc_cache["scores"]
                scores -= np.matrix(np.max(scores, axis=1)).T
                correct_score = scores[np.arange(batch_size), batch_Y]

                loss = -correct_score + np.log(np.sum(np.exp(scores), axis=1))
                loss = np.sum(loss)
                loss /= batch_size

                for layer in self.conv_W:
                    loss += 0.5 * self.reg * (np.sum(layer*layer))
                for layer in self.fc_layers:
                    loss += 0.5 * self.reg * (np.sum(layer*layer))
                loss += 0.5 * self.reg * (np.sum(self.output_W*self.output_W))

                if i%200 == 0:
                    print("Epoch %d ----- Iteration %d ----- Loss %f -----" % (e, i, loss))

                dout, d_cache = self._backward(out, batch_Y, fc_cache)

                dout = dout.reshape(flatten_shape)

                d_conv_cache = {}

                for l in reversed(range(len(self.conv_layers))):
                    layer = self.conv_layers[l]

                    if layer["name"] == "conv":
                        dout, _, _ = self._conv_backward(dout, conv_cache[l])
                    elif layer["name"] == "pool":
                        dout = self._pooling_backward(dout, conv_cache[l])

                self._update_weight(self.learning_rate, d_cache, d_conv_cache)
                    
if __name__ == '__main__':
    cnn = CNN(reg = 0.25, learning_rate = 0.24, std = 1e-2)
    cnn.load_data()
    cnn.train()
