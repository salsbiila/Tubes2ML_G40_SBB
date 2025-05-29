import numpy as np
from .base_layer import Layer

class Conv2DLayer(Layer):
    def __init__(self, num_filters, filter_size, stride=1, padding='valid', 
                 weight_initializer_mode='he', bias_initializer_mode='zeros'):
        super().__init__()
        if not isinstance(filter_size, tuple) or len(filter_size) != 2:
            raise ValueError("filter_size must be a tuple (height, width)")
        self.num_filters = num_filters
        self.f_h, self.f_w = filter_size

        if isinstance(stride, int):
            self.stride_h = self.stride_w = stride
        elif isinstance(stride, tuple) and len(stride) == 2:
            self.stride_h, self.stride_w = stride
        else:
            raise ValueError("stride must be an int or a tuple of 2 integers")

        self.padding_mode = padding.lower()
        self.weight_initializer_mode = weight_initializer_mode
        self.bias_initializer_mode = bias_initializer_mode
        
        self._input_channels = None
        self.pad_dims = (0,0,0,0)
        self.input_padded_for_backward = None
        self.input_tensor_original_shape = None

        # self.weights: (filter_height, filter_width, input_channels, num_filters)
        # self.biases: (1, 1, 1, num_filters) for broadcasting

    def initialize_parameters(self, input_channels):
        self._input_channels = input_channels
        
        # Weight Initialization
        if self.weight_initializer_mode == 'he':
            stddev = np.sqrt(2. / (self.f_h * self.f_w * self._input_channels))
            self.weights = np.random.randn(self.f_h, self.f_w, self._input_channels, self.num_filters) * stddev
        elif self.weight_initializer_mode == 'xavier':
            stddev = np.sqrt(2. / (self.f_h * self.f_w * self._input_channels + self.num_filters))
            self.weights = np.random.randn(self.f_h, self.f_w, self._input_channels, self.num_filters) * stddev
        elif self.weight_initializer_mode == 'zeros':
            self.weights = np.zeros((self.f_h, self.f_w, self._input_channels, self.num_filters))
        else: # Default to small random numbers
            print(f"Warning: Unknown weight_initializer_mode '{self.weight_initializer_mode}'. Using small random numbers.")
            self.weights = np.random.randn(self.f_h, self.f_w, self._input_channels, self.num_filters) * 0.01
        
        if self.bias_initializer_mode == 'zeros':
            self.biases = np.zeros((1, 1, 1, self.num_filters))
        else:
            print(f"Warning: Unknown bias_initializer_mode '{self.bias_initializer_mode}'. Using small random numbers for biases.")
            self.biases = np.random.randn(1, 1, 1, self.num_filters) * 0.01

    def forward(self, input_tensor):
        self.input_tensor_original_shape = input_tensor.shape 
        batch_size, i_h, i_w, i_c = input_tensor.shape
        
        if self.weights is None: # Initialize parameters on the first forward pass
            self.initialize_parameters(i_c)
        
        if self._input_channels != i_c:
            raise ValueError(f"Conv2DLayer input channel mismatch. Expected {self._input_channels}, got {i_c}.")

        if self.padding_mode == 'valid':
            self.pad_dims = (0,0,0,0)
            o_h = (i_h - self.f_h) // self.stride_h + 1
            o_w = (i_w - self.f_w) // self.stride_w + 1
            input_padded = input_tensor 
        elif self.padding_mode == 'same':
            o_h = int(np.ceil(float(i_h) / float(self.stride_h)))
            o_w = int(np.ceil(float(i_w) / float(self.stride_w)))
            
            pad_h_total = max((o_h - 1) * self.stride_h + self.f_h - i_h, 0)
            pad_w_total = max((o_w - 1) * self.stride_w + self.f_w - i_w, 0)

            pad_top = pad_h_total // 2
            pad_bottom = pad_h_total - pad_top
            pad_left = pad_w_total // 2
            pad_right = pad_w_total - pad_left
            self.pad_dims = (pad_top, pad_bottom, pad_left, pad_right)
            
            input_padded = np.pad(input_tensor, 
                                  ((0,0), (pad_top, pad_bottom), (pad_left, pad_right), (0,0)), 
                                  mode='constant', constant_values=0)
        else:
            raise ValueError(f"Unsupported padding mode: {self.padding_mode}. Choose 'valid' or 'same'.")
        
        self.input_padded_for_backward = input_padded

        output_tensor = np.zeros((batch_size, o_h, o_w, self.num_filters))

        for b in range(batch_size):
            for h_out in range(o_h):
                for w_out in range(o_w):
                    h_start = h_out * self.stride_h
                    h_end = h_start + self.f_h
                    w_start = w_out * self.stride_w
                    w_end = w_start + self.f_w
                    
                    receptive_field = input_padded[b, h_start:h_end, w_start:w_end, :]
                    
                    for f_idx in range(self.num_filters):
                        # Element-wise multiplication and sum, then add bias
                        # Weights for current filter: self.weights[:, :, :, f_idx]
                        # Bias for current filter: self.biases[0, 0, 0, f_idx]
                        conv_val = np.sum(receptive_field * self.weights[:, :, :, f_idx]) + self.biases[0, 0, 0, f_idx]
                        output_tensor[b, h_out, w_out, f_idx] = conv_val
        
        self.output = output_tensor
        return self.output

    def backward(self, output_gradient):
        batch_size, o_h, o_w, _ = output_gradient.shape
        
        self.d_weights = np.zeros_like(self.weights)
        self.d_biases = np.zeros_like(self.biases)
        d_input_padded = np.zeros_like(self.input_padded_for_backward)

        for b in range(batch_size):
            for h_out in range(o_h):
                for w_out in range(o_w):
                    # Define the receptive field in the (padded) input corresponding to this output unit
                    h_start = h_out * self.stride_h
                    h_end = h_start + self.f_h
                    w_start = w_out * self.stride_w
                    w_end = w_start + self.f_w
                    
                    receptive_field_padded = self.input_padded_for_backward[b, h_start:h_end, w_start:w_end, :]
                    
                    for f_idx in range(self.num_filters):
                        # Gradient dL/dO for the current output unit O[b, h_out, w_out, f_idx]
                        grad_slice_output = output_gradient[b, h_out, w_out, f_idx]
                        
                        # Gradient for weights dL/dW_f:
                        # dL/dW_f += (dL/dO_current * X_receptive_field)
                        self.d_weights[:, :, :, f_idx] += receptive_field_padded * grad_slice_output
                        
                        # Gradient for biases dL/dB_f:
                        # dL/dB_f += dL/dO_current
                        self.d_biases[0, 0, 0, f_idx] += grad_slice_output
                        
                        # Gradient for the padded input dL/dX_padded:
                        # dL/dX_receptive_field += (dL/dO_current * W_f)
                        d_input_padded[b, h_start:h_end, w_start:w_end, :] += self.weights[:, :, :, f_idx] * grad_slice_output
        
        # If 'same' padding was used, remove the padding from d_input_padded to get dL/dI_original
        if self.padding_mode == 'same':
            pad_top, pad_bottom, pad_left, pad_right = self.pad_dims
            # Extract the central part corresponding to the original input dimensions
            # Original height and width from stored shape
            orig_h = self.input_tensor_original_shape[1]
            orig_w = self.input_tensor_original_shape[2]
            
            d_input = d_input_padded[:, pad_top : pad_top + orig_h, pad_left : pad_left + orig_w, :]
        else: # 'valid' padding, no padding was added to input
            d_input = d_input_padded
            
        return d_input
