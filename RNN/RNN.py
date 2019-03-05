import numpy as np


class RNN:
    """ A RNN modal 

    [public method]
    train --- training the model to get the better weight
    
    """
    def __init__(self, hidden_size,learning_rate, filename):
        self.data = None

        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.chars_list = []
        self.vocab_size = None
        self.data_size = None

        self.Wxh = None
        self.Whh = None
        self.Why = None
        self.bh = None
        self.by = None

        self.mWxh = None
        self.mWhh = None
        self.mWhy = None
        self.mbh = None
        self.mby = None

        self.vWxh = None
        self.vWhh = None
        self.vWhy = None
        self.vbh = None
        self.vby = None

        self.load_data(filename)

    def _init_weight(self):
        """initialize the weight to random scale
        """
        self.Wxh = np.random.randn(self.hidden_size, self.vocab_size)*0.01
        self.Whh = np.random.randn(self.hidden_size, self.hidden_size)*0.01
        self.Why = np.random.randn(self.vocab_size, self.hidden_size)*0.01

        self.bh = np.zeros((self.hidden_size, 1))
        self.by = np.zeros((self.vocab_size, 1))
        
        self.mWxh, self.mWhh, self.mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
        self.mbh, self.mby = np.zeros_like(bh), np.zeros_like(by)

    def load_data(self, filename):
        """load a txt file 
        """
        data = open(filename, 'r').read()

        self.chars_list = list(set(data))
        self.data_size = len(data)
        self.vocab_size = len(self.chars_list)

        self.data = data

    def train(self, batch_size, epochs):
        """train RNN
        """
        iterations = self.data_size//batch_size 

        for e in range(epochs):
            for i in range(iterations):

                batch_data = self.data[i*batch_size, (i+1)*batch_size]
                batch_target = self.data[i*batch_size+1, (i+1)*batch_size+1]]

                x_state, h_state, y_state, predict_state = {}, {}, {}, {}
                h_state[-1] = np.copy(hprev)

                loss = 0

                # forward pass
                for b in range(batch_size):
                    x_state[b] = np.zeros((self.vocab_size, 1))
                    x_state[b][batch_data[b]] = 1

                    h_state[b] = np.tanh(np.dot(self.Wxh, x_state[b])+np.dot(self.Whh, h_state[b-1])+self.bh)
                    y_state[b] = np.dot(self.Why, h_state[b]) + self.by

                    predict_state[b] = np.exp(y_state[b]) / np.sum(np.exp(y_state[b]))
                    loss += -np.log(predict_state[b][batch_target[b]])

                dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
                dbh, dby = np.zeros_like(bh), np.zeros_like(by)
                dhnext = np.zeros((self.hidden_size, 1))

                for b in reversed(range(batch_size)):
                    # Softmax gradient, see https://raw.githubusercontent.com/eric8607242/cs231n_2017_assignment/master/assignment1/softmax_gradient.jpeg to know more detail
                    dloss = np.copy(predict_state[b])
                    dloss[batch_target[b]] += -1

                    dWhy += np.dot(dloss, h_state[b].T)
                    dby += dloss 

                    dh = np.dot(Why.T, dloss) + dhnext

                    # dtanh(x)/dx = sech(x)^2
                    # tanh^2(x) + sech^2(x) = 1
                    # dtanh = 1 - tanh^2(x)
                    dtanh = (1 - h_state[b] * h_state[b]) * dh

                    dWhh += np.dot(dtanh, h_state[b-1].T)
                    dbh += dtanh

                    dWxh += np.dot(dtanh, x_state[b].T)
                    dhnext = np.dot(Whh.T, dtanh)

                for dparam in [dWxh, dWhh, dWhy, dbh, dby, h_state[len(batch_data)-1]
                    np.clip(dparam, -5, 5, out=dparam)
                
                dparam_list = [dWxh, dWhh, dWhy, dbh, dby]
                self._update_weight(dparam_list)

                 
    def _update_weight(self, dparam_list):

        beta1, beta2, eps = 0.9, 0.999, 1e-8
        param_list = [self.Wxh, self.Whh, self.Why, self.bh, self.by]
        mparam_list = [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby]
        vparam_list = [self.vWxh, self.vWhh, self.vWhy, self.vbh, self.vby]

        for param, dparam, mem, v in zip(param_list, dparam_list, mparam_list, vparam_list):
            mem += beta1 * mem + (1-beta1)*dparam
            v = beta2 * v + (1-beta2)*dparam*dparam
            param += -self.learning_rate * mem / (np.sqrt(v+eps))
                    
    def _chars_to_index(self, chars):
        return {char:index for index, cahr in enumerate(chars)}
        
    def predict(self):
    
