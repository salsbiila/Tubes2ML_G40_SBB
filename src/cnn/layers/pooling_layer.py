import numpy as np
from .base_layer import Layer

class PoolingLayer(Layer):
    def __init__(self, pool_size=(2,2), stride=None, mode='max'):
        super().__init__()
        if not isinstance(pool_size, tuple) or len(pool_size) != 2:
            raise ValueError("pool_size must be a tuple of 2 integers (height, width)")
        self.pool_h, self.pool_w = pool_size
        
        self.stride_h = stride if stride is not None else self.pool_h
        self.stride_w = stride if stride is not None else self.pool_w
        if isinstance(stride, tuple) and len(stride) == 2:
            self.stride_h, self.stride_w = stride
        elif isinstance(stride, int):
            self.stride_h = self.stride_w = stride
        
        self.mode = mode.lower()
        self.cache = {}

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size, i_h, i_w, i_c = input_tensor.shape

        o_h = (i_h - self.pool_h) // self.stride_h + 1
        o_w = (i_w - self.pool_w) // self.stride_w + 1
        
        output_tensor = np.zeros((batch_size, o_h, o_w, i_c))
        self.cache = {}

        for b in range(batch_size):
            for c_idx in range(i_c):
                for h_out in range(o_h):
                    for w_out in range(o_w):
                        h_start = h_out * self.stride_h
                        h_end = h_start + self.pool_h
                        w_start = w_out * self.stride_w
                        w_end = w_start + self.pool_w
                        
                        receptive_field = input_tensor[b, h_start:h_end, w_start:w_end, c_idx]
                        
                        if self.mode == 'max':
                            output_tensor[b, h_out, w_out, c_idx] = np.max(receptive_field)
                            max_idx_local = np.unravel_index(np.argmax(receptive_field), receptive_field.shape)
                            self.cache[(b, h_out, w_out, c_idx)] = (h_start + max_idx_local[0], w_start + max_idx_local[1])
                        elif self.mode == 'average':
                            output_tensor[b, h_out, w_out, c_idx] = np.mean(receptive_field)
                        else:
                            raise ValueError(f"Unsupported pooling mode: {self.mode}. Choose 'max' or 'average'.")
        self.output = output_tensor
        return self.output

    def backward(self, output_gradient):
        d_input = np.zeros_like(self.input_tensor)
        batch_size, o_h, o_w, i_c = output_gradient.shape

        for b in range(batch_size):
            for c_idx in range(i_c):
                for h_out in range(o_h):
                    for w_out in range(o_w):
                        grad_val = output_gradient[b, h_out, w_out, c_idx]
                        
                        if self.mode == 'max':
                            # Distribute gradient only to the position of the max value
                            # from the forward pass cache
                            h_idx_in, w_idx_in = self.cache[(b, h_out, w_out, c_idx)]
                            d_input[b, h_idx_in, w_idx_in, c_idx] += grad_val # Use += for safety if pools overlap
                        elif self.mode == 'average':
                            # Distribute gradient equally to all elements in the receptive field
                            h_start = h_out * self.stride_h
                            h_end = h_start + self.pool_h
                            w_start = w_out * self.stride_w
                            w_end = w_start + self.pool_w
                            
                            # The gradient is divided by the number of elements in the pool window
                            avg_grad = grad_val / (self.pool_h * self.pool_w)
                            d_input[b, h_start:h_end, w_start:w_end, c_idx] += avg_grad
        return d_input