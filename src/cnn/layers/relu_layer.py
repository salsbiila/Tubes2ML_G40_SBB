import numpy as np
from .base_layer import Layer

class ReLULayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.output = np.maximum(0, input_tensor)
        return self.output

    def backward(self, output_gradient):
        # 1 if x > 0, and 0 if x <= 0.
        # dL/dI = dL/dO * dO/dI
        d_input = output_gradient * (self.input_tensor > 0)
        return d_input