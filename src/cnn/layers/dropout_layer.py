# src/cnn/layers/dropout_layer.py
import numpy as np
from .base_layer import Layer

class DropoutLayer(Layer):
    def __init__(self, rate):
        super().__init__()
        if not 0 <= rate < 1: # Dropout rate is probability of setting unit to 0
            raise ValueError("Dropout rate must be in the interval [0, 1).")
        self.rate = rate 
        self.keep_prob = 1.0 - self.rate
        self.mask = None # To store the dropout mask for use in backward pass

    def forward(self, input_tensor):
        self.input_tensor = input_tensor # Store for potential checks, not strictly for backward math
        
        if self.training_mode:
            # Inverted Dropout:
            # 1. Generate a mask with 0s (for dropped units) and 1s (for kept units).
            #    Probability of 1 is keep_prob.
            # 2. Scale the kept units by 1/keep_prob.
            # This way, during inference, no scaling is needed.
            self.mask = (np.random.rand(*input_tensor.shape) < self.keep_prob) / self.keep_prob
            self.output = input_tensor * self.mask
        else: # Inference mode
            # During inference, dropout is turned off, and the output is simply the input.
            # No scaling is needed because of the inverted dropout applied during training.
            self.output = input_tensor 
            self.mask = None # Mask is not generated or used during inference
            
        return self.output

    def backward(self, output_gradient):
        """
        Computes the gradient of the loss w.r.t. the input of Dropout.
        output_gradient: numpy array, dL/dO.
        Returns: numpy array, dL/dI.
        """
        # The gradient is passed only through the units that were not dropped out (where mask > 0).
        # The scaling applied during the forward pass (1/keep_prob) is also applied to the gradient.
        if not self.training_mode:
            # If called in inference mode (should ideally not happen for backprop),
            # or if forward pass was in inference mode, self.mask is None.
            # In this case, gradient passes through unchanged.
            return output_gradient

        if self.mask is None:
            # This state should ideally not be reached if backward is called after a training forward pass.
            # It implies something went wrong, or backward is being called inappropriately.
            raise RuntimeError("Dropout backward called in training mode, but mask is None. "
                               "Ensure forward pass in training mode was called first.")

        d_input = output_gradient * self.mask
        return d_input
