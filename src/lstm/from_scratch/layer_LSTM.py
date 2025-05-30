import numpy as np

class LSTMCell: 
    def __init__(self, kernel, recurrent_kernel, bias, use_bias=True):
        self.kernel = kernel
        self.recurrent_kernel = recurrent_kernel
        self.bias = bias if use_bias else None
        self.use_bias = use_bias

        self.input_dim = kernel.shape[0]
        self.units = recurrent_kernel.shape[0]

        self.W_i = kernel[:, :self.units]                   
        self.W_f = kernel[:, self.units:2*self.units]        
        self.W_c = kernel[:, 2*self.units:3*self.units]      
        self.W_o = kernel[:, 3*self.units:] 

        self.U_i = recurrent_kernel[:, :self.units]
        self.U_f = recurrent_kernel[:, self.units:2*self.units]
        self.U_c = recurrent_kernel[:, 2*self.units:3*self.units]
        self.U_o = recurrent_kernel[:, 3*self.units:]

        if self.use_bias and bias is not None:
            self.b_i = bias[:self.units]
            self.b_f = bias[self.units:2*self.units]
            self.b_c = bias[2*self.units:3*self.units]
            self.b_o = bias[3*self.units:]
        else:
            self.b_i = self.b_f = self.b_c = self.b_o = None

    def forward_step(self, x_t, states):
        h_prev, c_prev = states

        i_t = np.dot(x_t, self.W_i) + np.dot(h_prev, self.U_i)
        if self.b_i is not None:
            i_t += self.b_i
        i_t = self._sigmoid(i_t)

        f_t = np.dot(x_t, self.W_f) + np.dot(h_prev, self.U_f)
        if self.b_f is not None:
            f_t += self.b_f
        f_t = self._sigmoid(f_t)

        candidate_t = np.dot(x_t, self.W_c) + np.dot(h_prev, self.U_c)
        if self.b_c is not None:
            candidate_t += self.b_c
        candidate_t = np.tanh(candidate_t)

        c_t = f_t * c_prev + i_t * candidate_t

        o_t = np.dot(x_t, self.W_o) + np.dot(h_prev, self.U_o)
        if self.b_o is not None:
            o_t += self.b_o
        o_t = self._sigmoid(o_t)

        h_t = o_t * np.tanh(c_t)
        
        return h_t, c_t
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
class LSTMLayer:
    def __init__(self, weights, return_sequences=False, go_backwards=False):
        if len(weights) == 3:
            kernel, recurrent_kernel, bias = weights
            self.cell = LSTMCell(kernel, recurrent_kernel, bias, use_bias=True)
        elif len(weights) == 2:
            kernel, recurrent_kernel = weights
            self.cell = LSTMCell(kernel, recurrent_kernel, None, use_bias=False)
        else:
            raise ValueError(f"Expected 2 or 3 weight arrays, got {len(weights)}")
        
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards

    def forward(self, x):
        original_shape = x.shape
        
        if len(x.shape) == 1:
            x = x.reshape(1, 1, -1)
        elif len(x.shape) == 2:
            x = x.reshape(x.shape[0], 1, x.shape[1])
        elif len(x.shape) == 3:
            pass
        else:
            raise ValueError(f"Input must have 1, 2, or 3 dimensions, got {len(x.shape)} dimensions with shape {x.shape}")
        
        batch_size, sequence_length, input_dim = x.shape

        if self.go_backwards:
            x = x[:, ::-1, :]

        h_t = np.zeros((batch_size, self.cell.units))
        c_t = np.zeros((batch_size, self.cell.units))

        hidden_states = []

        for t in range(sequence_length):
            x_t = x[:, t, :]
            h_t, c_t = self.cell.forward_step(x_t, (h_t, c_t))
            hidden_states.append(h_t.copy())

        if self.return_sequences:
            output = np.stack(hidden_states, axis=1)

            if self.go_backwards:
                output = output[:, ::-1, :]
            return output
        else :
            if len(original_shape) == 1:
                return hidden_states[-1].squeeze(0)
            else:
                return hidden_states[-1]