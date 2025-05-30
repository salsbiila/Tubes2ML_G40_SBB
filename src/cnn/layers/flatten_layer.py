import numpy as np
from .base_layer import Layer

import numpy as np
from .base_layer import Layer

class FlattenLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.original_shape = input_tensor.shape 
        
        batch_size = input_tensor.shape[0]
        self.output = input_tensor.reshape(batch_size, -1) 
        return self.output

    def backward(self, output_gradient):
        d_input = output_gradient.reshape(self.original_shape)
        return d_input