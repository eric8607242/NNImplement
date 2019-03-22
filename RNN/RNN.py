import numpy as np

class RNN:
    def __init__(self, hidden_size, learning_rate):
        """Initialize the RNN model and assign None value to the variable

        Args:
            hidden_size(int):hidden size for RNN model

        Attributes:
            data(str):training data
            char_to_index(dict):the mapping between relationship the each char and each index in the chars set 
            index_to_char(dict):the mapping between relationship the each index and each char in the chars set
            data_size(int):size of the training data
            chars_size(int):the size of the chars set which is the total chars in training data without repeat
            learning_rate(float):learning rate in training step
            hidden_size(int):hidden size for RNN model
            Wxh, Whh, Why(ndarray):the weight for training step
            bh, by(ndarray):the bias for trainging step
        """
        self.data = None
        self.char_to_index = None
        self.index_to_char = None
        self.data_size = None
        self.chars_size = None

        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.hprev = None

        self.Wxh, self.Whh, self.Why = None, None, None
        self.bh, self.by = None, None

    def _init_weight(self):
        """Initialize the weight value accrodding for the size of the data and hidden size """
        self.Wxh, self.Whh, self.Why = (
               np.random.rand(self.hidden_size, self.chars_size)*0.01,
               np.random.rand(self.hidden_size, self.hidden_size)*0.01,
               np.random.rand(self.chars_size, self.hidden_size)*0.01,
            )
        self.bh, self.by = (
               np.zeros((self.hidden_size, 1)),
               np.zeros((self.chars_size, 1))
            )
        
    def load_data(self, filename):
        """load the data via the path of the input data

        Args:
            filename(str):the path of the input data
        """
        data = open(filename, 'r').read()

        chars_set = list(set(data))
        self.char_to_index = {ch:i for i,ch in enumerate(chars_set)}
        self.index_to_char = {i:ch for i,ch in enumerate(chars_set)}

        self.data_size = len(data)
        self.chars_size = len(chars_set)

        self._init_weight();

        self.data = data


    def train(self, batch_size, epochs):
        """training model with SGD"""
        print("---training---")

        iterations = self.data_size//batch_size
        hprev = np.zeros((self.hidden_size, 1))

        for e in range(epochs):
            for i in range(iterations):
                # generate input data and label with batch_size
                b_data = self.data[i*batch_size:(i+1)*batch_size]
                b_data_ch_id = self._chars_to_index(b_data)

                b_label = self.data[i*batch_size+1:(i+1)*batch_size+1]
                b_label_ch_id = self._chars_to_index(b_label)

                state_cache, loss = self._forward(batch_size, hprev, b_data_ch_id, b_label_ch_id)
                
                if i % 100 == 0:
                    print("Epoach %d Iteration %d -----Loss %d" % (e, i, loss))

                dWxh, dWhh, dbh, dWhy, dby = self._backward(batch_size, state_cache, b_label_ch_id)

                hprev = state_cache[1][batch_size-1]
                for dparam in [dWxh, dWhh, dbh, dWhy, dby]:
                    np.clip(dparam, -1, 1, out=dparam)
                
                self._update_weight([dWxh, dWhh, dbh, dWhy, dby])

            self.hprev = hprev
            self.predict(165, self.data[0])
        

    def _forward(self, batch_size, hprev, b_data_ch_id, b_label_ch_id):
        """forwarding
        Args:
            hprev(ndarray):the previous h_state in the last training step 
            b_data_ch_id(list): the index of the batch data in char set of the input data
            b_label_ch_id(list): the index of the batch label in char set of the input data

        Return:
            x_state(dict):the each char state in batch data
            h_state(dict):the each h state with each of the batch data
            y_state(dict):the each y state with each of the batch data
            predice_state(dict):the each predict result with each of the batch data
            loss(ndarray):the total loss for this forwarding
        """
        x_state, h_state, y_state, predict_state = {}, {}, {}, {}
        h_state[-1] = hprev.copy()
        loss = 0

        for b in range(batch_size):
            hprev = h_state[b-1]
            # generate input data matrix
            x_state[b] = np.zeros((self.chars_size, 1))
            x_state[b][b_data_ch_id[b]] = 1

            h_state[b] = np.tanh(np.dot(self.Wxh, x_state[b])+np.dot(self.Whh, hprev))+self.bh
            y_state[b] = np.dot(self.Why, h_state[b])+self.by
            predict_state[b] = np.exp(y_state[b]) / np.sum(np.exp(y_state[b]))
            loss += -np.log(predict_state[b][b_label_ch_id[b]])

        return (x_state, h_state, y_state, predict_state),  loss

    def _backward(self, batch_size, state_cache, b_label_ch_id):
        """backwarding

        Args:
            state_cache(tuple):all state in the forwarding step
            b_label_ch_id(list):the index of the batch label in char set of the input data

        Returns:
            (ndarray):the gradient of each weight

        """
        x_state, h_state, y_state, predict_state = state_cache

        dWxh, dWhh, dWhy = (
               np.zeros_like(self.Wxh),
               np.zeros_like(self.Whh),
               np.zeros_like(self.Why)
            ) 
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros((self.hidden_size, 1))

        for b in reversed(range(batch_size)):
            # if you want to know why the gradient of Softmax is.
            # you can check https://reurl.cc/Gpr7y
            dloss = np.copy(predict_state[b])
            dloss[b_label_ch_id[b]] += -1

            dWhy += np.dot(dloss, h_state[b].T)
            dby += dloss

            dh_state = np.dot(self.Why.T, dloss) + dhnext

            # dtanh(x)/dx = secch(x)^2
            # tanh^2(x) + sech^2(x) = 1
            # dtanh = 1 - tanh^2(x)
            dtanh = (1-h_state[b] * h_state[b]) * dh_state
            
            dWhh += np.dot(dtanh, h_state[b-1].T)
            dbh += dtanh

            dWxh += np.dot(dtanh, x_state[b].T)
            dhnext = np.dot(self.Whh.T, dtanh)
        
        return dWxh, dWhh, dbh, dWhy, dby
            
    def _update_weight(self, dparam_list):
        """use SGD to update weight with gradient of each weight

        Args:
            dparam_list(list): the list that contains the gradient of each weight
        """
        param_list = [self.Wxh, self.Whh, self.bh, self.Why, self.by]
        
        for param, dparam in zip(param_list, dparam_list):
            param += -self.learning_rate * dparam 
            
    def _chars_to_index(self, chars):
        """convert a string of characters to the index in the character set

        Args:
            chars(string):the string will be convert to index

        Returns:
            (list):the list that each character in chars is mapped to the index in character set
        """
        return [self.char_to_index[ch] for ch in chars]

    def _index_to_chars(self, index):
        """convert a list of integer which is the index of the character to the character in the character set

        Args:
            index(list):the integer list will be convert to character

        Returns:
            (list):the list that each integer in index is mapped to the character in character set
        """
        return [self.index_to_char[i] for i in index]
    
    def predict(self, seq_len, input_data):
        """predict the sequence 

        Args:
            seq_len(int): the length that predict string will be
            input_data(char): the first char to feed into RNN model
        """
        # generate the input data matrix
        x = np.zeros((self.chars_size, 1))
        x_id = self._chars_to_index(input_data)
        x[x_id] = 1
        h_state = self.hprev        

        seq_index = []
        for t in range(seq_len):
            h_state = np.tanh(np.dot(self.Wxh, x)+np.dot(self.Whh, h_state)) + self.bh
            y_state = np.dot(self.Why, h_state) + self.by
            p_state = np.exp(y_state)/np.sum(np.exp(y_state))

            x_id = np.argmax(p_state)
            x = np.zeros((self.chars_size, 1))
            x[x_id] = 1
            seq_index.append(x_id)

        output_seq = ''.join(self._index_to_chars(seq_index))
        print(output_seq)

if __name__=='__main__':
    rnn = RNN(200, 3e-3);
    rnn.load_data('input.txt')
    rnn.train(165,50)
