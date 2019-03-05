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
        
        self.chars_dict = {}
        self.index_dict = {}

        self.vocab_size = None
        self.data_size = None

        self.hprev = None

        self.Wxh, self.Whh, self.Why = None, None, None
        self.bh, self.by = None, None

        self.mWxh, self.mWhh, self.mWhy = None, None, None
        self.mbh, self.mby = None, None

        self.vWxh, self.vWhh, self.vWhy = None, None, None
        self.vbh, self.vby = None, None

        self.load_data(filename)

    def _init_weight(self):
        """initialize the weight to random scale
        """
        self.Wxh = np.random.randn(self.hidden_size, self.vocab_size)*0.01
        self.Whh = np.random.randn(self.hidden_size, self.hidden_size)*0.01
        self.Why = np.random.randn(self.vocab_size, self.hidden_size)*0.01

        self.bh = np.zeros((self.hidden_size, 1))
        self.by = np.zeros((self.vocab_size, 1))
        
        self.mWxh, self.mWhh, self.mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        self.mbh, self.mby = np.zeros_like(self.bh), np.zeros_like(self.by)

        self.vWxh, self.vWhh, self.vWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        self.vbh, self.vby = np.zeros_like(self.bh), np.zeros_like(self.by)

    def load_data(self, filename):
        """load a txt file 
        """
        print("--load data---")
        data = open(filename, 'r').read()

        chars_set = list(set(data))
        for i, ch in enumerate(chars_set):
            self.chars_dict[ch] = i
            self.index_dict[i] = ch
        # self.chars_dict = {ch:i for i,ch in enumerate(chars_set)}
        # self,index_dict = {i:ch for i,ch in enumerate(chars_set)}
        self.data_size = len(data)
        self.vocab_size = len(chars_set)

        self.hprev = np.zeros((self.hidden_size, 1))
        self.data = data

        self._init_weight()

    def train(self, batch_size, epochs):
        """train RNN
        """
        print("---training start---")
        iterations = self.data_size//batch_size 

        for e in range(epochs):
            print("---epochs %d start---" % e)
            hprev = np.zeros_like(self.hprev)
            for i in range(iterations):
                batch_data = self.data[i*batch_size:(i+1)*batch_size]
                batch_ch_id = self._chars_to_index(batch_data)
                batch_target = self.data[i*batch_size+1:(i+1)*batch_size+1]
                batch_target_ch_id = self._chars_to_index(batch_target)

                x_state, h_state, y_state, predict_state = {}, {}, {}, {}
                print(hprev)
                h_state[-1] = np.copy(hprev)
                loss = 0

                # forward pass
                for b in range(batch_size-1):
                    x_state[b] = np.zeros((self.vocab_size, 1))
                    x_state[b][batch_ch_id[b]] = 1

                    h_state[b] = np.tanh(np.dot(self.Wxh, x_state[b])+np.dot(self.Whh, h_state[b-1])+self.bh)
                    y_state[b] = np.dot(self.Why, h_state[b]) + self.by

                    predict_state[b] = np.exp(y_state[b]) / np.sum(np.exp(y_state[b]))
                    loss += -np.log(predict_state[b][batch_target_ch_id[b]]) 
                    dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
                dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
                dhnext = np.zeros((self.hidden_size, 1))

                for b in reversed(range(batch_size-1)):
                    # Softmax gradient, see https://raw.githubusercontent.com/eric8607242/cs231n_2017_assignment/master/assignment1/softmax_gradient.jpeg to know more detail
                    dloss = np.copy(predict_state[b])
                    dloss[batch_target_ch_id[b]] += -1

                    dWhy += np.dot(dloss, h_state[b].T)
                    dby += dloss 

                    dh = np.dot(self.Why.T, dloss) + dhnext

                    # dtanh(x)/dx = sech(x)^2
                    # tanh^2(x) + sech^2(x) = 1
                    # dtanh = 1 - tanh^2(x)
                    dtanh = (1 - h_state[b] * h_state[b]) * dh

                    dWhh += np.dot(dtanh, h_state[b-1].T)
                    dbh += dtanh

                    dWxh += np.dot(dtanh, x_state[b].T)
                    dhnext = np.dot(self.Whh.T, dtanh)

                for dparam in [dWxh, dWhh, dWhy, dbh, dby, h_state[len(batch_data)-2]]:
                    np.clip(dparam, -1, 1, out=dparam)
                
                dparam_list = [dWxh, dWhh, dWhy, dbh, dby]
                hprev = h_state[len(batch_data)-2]
                self._update_weight(dparam_list)
            self.hprev = hprev
                 
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
        return [self.chars_dict[ch] for ch in chars]

    def _index_to_chars(self, index):
        return [self.index_dict[i] for i in index]

    def predict(self, seq_len, input_data):
        x = np.zeros((self.vocab_size, 1))
        x_id = self._chars_to_index(input_data)
        x[x_id] = 1
        h = np.copy(self.hprev)

        seq_index = []
        for t in range(seq_len):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh) 
            y = np.dot(self.Why, h) + self.by
            p = np.exp(y) / np.sum(np.exp(y))
            
            x_id = np.argmax(p)
            x = np.zeros((self.vocab_size, 1))
            x[x_id] = 1
            seq_index.append(x_id)

        output_seq = ''.join(self._index_to_chars(seq_index))
        print(output_seq)
            

if __name__ == '__main__' :
    rnn = RNN(100, 1e-3, 'input.txt')
    rnn.train(3, 2)
    rnn.predict(10, "h")
