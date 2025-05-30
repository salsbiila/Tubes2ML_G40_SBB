import numpy as np
from .base_layer import Layer

class DenseLayer(Layer):
    def __init__(self, output_dim, weight_initializer_mode='he', bias_initializer_mode='zeros'):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = None
        self.weight_initializer_mode = weight_initializer_mode
        self.bias_initializer_mode = bias_initializer_mode

    def initialize_parameters(self, input_dim_val):
        if self.input_dim is not None and self.input_dim != input_dim_val:
            print(f"Warning: Re-initializing DenseLayer with new input_dim {input_dim_val} (was {self.input_dim})")

        self.input_dim = input_dim_val
        
        if self.weight_initializer_mode == 'he':
            self.weights = np.random.randn(self.input_dim, self.output_dim) * np.sqrt(2. / self.input_dim)
        elif self.weight_initializer_mode == 'xavier':
            self.weights = np.random.randn(self.input_dim, self.output_dim) * np.sqrt(1. / self.input_dim)
        elif self.weight_initializer_mode == 'zeros':
             self.weights = np.zeros((self.input_dim, self.output_dim))
        else: 
            print(f"Warning: Unknown weight_initializer_mode '{self.weight_initializer_mode}'. Using small random numbers.")
            self.weights = np.random.randn(self.input_dim, self.output_dim) * 0.01

        if self.bias_initializer_mode == 'zeros':
            self.biases = np.zeros((1, self.output_dim))
        else:
            print(f"Warning: Unknown bias_initializer_mode '{self.bias_initializer_mode}'. Using small random numbers for biases.")
            self.biases = np.random.randn(1, self.output_dim) * 0.01

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        
        if self.weights is None:
            self.initialize_parameters(input_tensor.shape[1])
        elif self.input_dim != input_tensor.shape[1]:
             raise ValueError(f"DenseLayer input dimension mismatch. Expected {self.input_dim}, got {input_tensor.shape[1]}. Re-initialization might be needed.")

        # Linear transformation: Output = Input @ Weights + Biases
        self.output = np.dot(input_tensor, self.weights) + self.biases
        return self.output

    def backward(self, output_gradient):
        # dL/dW = dL/dO * dO/dW 
        # dO/dW = X^T (transpose of input_tensor)
        # dL/dW = X^T @ dL/dO
        self.d_weights = np.dot(self.input_tensor.T, output_gradient)
        
        # dL/dB = dL/dO * dO/dB
        # dO/dB = 1 (for each element in the batch)
        # So, dL/dB = sum(dL/dO) over the batch dimension.
        # Shape: sum over axis 0 of (batch_size, output_dim) -> (1, output_dim)
        self.d_biases = np.sum(output_gradient, axis=0, keepdims=True)
        
        # dL/dI = dL/dO * dO/dI
        # dO/dI = W^T (transpose of weights)
        # dL/dI = dL/dO @ W^T
        d_input = np.dot(output_gradient, self.weights.T)
        
        return d_input
