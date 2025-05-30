import numpy as np

class SimpleRNNCell:
    def __init__(self, kernel, recurrent_kernel, bias, use_bias=True):
        self.kernel = kernel
        self.recurrent_kernel = recurrent_kernel
        self.bias = bias if use_bias else None
        self.units = recurrent_kernel.shape[0]
        self.use_bias = use_bias
    
    def forward_step(self, x_t, h_prev):
        # h_t = tanh(x_t @ W_input + h_prev @ W_recurrent + bias)
        linear = np.dot(x_t, self.kernel) + np.dot(h_prev, self.recurrent_kernel)
        if self.use_bias and self.bias is not None:
            linear += self.bias
        h_t = np.tanh(linear)
        return h_t

class SimpleRNNLayer:
    def __init__(self, weights, return_sequences=False, go_backwards=False):
        if len(weights) == 3:
            kernel, recurrent_kernel, bias = weights
            self.cell = SimpleRNNCell(kernel, recurrent_kernel, bias, use_bias=True)
        elif len(weights) == 2:
            kernel, recurrent_kernel = weights
            self.cell = SimpleRNNCell(kernel, recurrent_kernel, None, use_bias=False)
        else:
            raise ValueError(f"Expected 2 or 3 weight arrays, got {len(weights)}")
        
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards
    
    def forward(self, x):
        # Handle different input shapes
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
        
        hidden_states = []
        h_t = np.zeros((batch_size, self.cell.units))

        for t in range(sequence_length):
            x_t = x[:, t, :]
            h_t = self.cell.forward_step(x_t, h_t)
            hidden_states.append(h_t.copy())
        
        if self.return_sequences:
            output = np.stack(hidden_states, axis=1)
            if self.go_backwards:
                output = output[:, ::-1, :]
            return output
        else:
            if len(original_shape) == 1:
                return hidden_states[-1].squeeze(0) 
            else:
                return hidden_states[-1]