# src/cnn/layers/flatten_layer.py
import numpy as np
from .base_layer import Layer

import numpy as np
from .base_layer import Layer

class FlattenLayer(Layer):
    def __init__(self):
        super().__init__()
        self.original_shape = None # To store the shape before flattening

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.original_shape = input_tensor.shape 
        
        batch_size = input_tensor.shape[0]
        # The '-1' infers the product of all other dimensions (H*W*C)
        self.output = input_tensor.reshape(batch_size, -1) 
        return self.output

    def backward(self, output_gradient):
        if self.original_shape is None:
            raise ValueError("Original shape not stored. Forward pass must be called before backward.")
        
        d_input = output_gradient.reshape(self.original_shape)
        return d_input