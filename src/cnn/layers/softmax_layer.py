import numpy as np
from .base_layer import Layer

class SoftmaxLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        exp_values = np.exp(input_tensor - np.max(input_tensor, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True) # Probabilities
        return self.output
    
    def backward(self, output_gradient):
        # calculation done in loss function
        return output_gradient
