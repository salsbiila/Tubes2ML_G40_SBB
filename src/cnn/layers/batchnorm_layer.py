# src/cnn/layers/batch_norm_layer.py
import numpy as np
from .base_layer import Layer

class BatchNormalizationLayer(Layer):
    def __init__(self, epsilon=1e-5, momentum=0.9):
        super().__init__()
        self.epsilon = epsilon  # To prevent division by zero
        self.momentum = momentum # For updating moving averages

        # Learnable parameters: gamma (scale) and beta (shift)
        self.gamma = None 
        self.beta = None  
        
        # Gradients for learnable parameters
        self.d_gamma = None
        self.d_beta = None

        # Moving averages for inference mode (non-learnable state)
        # These are updated during training using an exponential moving average
        self.moving_mean = None
        self.moving_variance = None

        # Intermediate values stored during forward pass for use in backward pass
        self.x_normalized = None    # Input normalized (before scaling by gamma)
        self.batch_mean = None      # Mean of the current batch
        self.batch_variance = None  # Variance of the current batch
        self.input_shape = None     # Shape of the input tensor
        self.inv_stddev = None      # 1 / sqrt(variance + epsilon)

    def initialize_parameters(self, num_features):
        # num_features is the number of channels for Conv2D output, 
        # or the number of features for Dense layer output.
        # Gamma and Beta are per-feature.
        
        # Shape for gamma and beta depends on whether input is from Conv or Dense
        # For Conv (4D input: B, H, W, C), params are per channel (C)
        # For Dense (2D input: B, D), params are per feature (D)
        if len(self.input_shape) == 4: # Convolutional features
            param_shape = (1, 1, 1, num_features)
        elif len(self.input_shape) == 2: # Dense features
            param_shape = (1, num_features)
        else:
            raise ValueError(f"Unsupported input shape for BatchNormalization: {self.input_shape}")

        self.gamma = np.ones(param_shape)
        self.beta = np.zeros(param_shape)
        
        # Initialize moving averages
        self.moving_mean = np.zeros(param_shape)
        self.moving_variance = np.ones(param_shape) # Initialize to ones, not zeros, for stability

    def forward(self, input_tensor):
        self.input_tensor = input_tensor # Store for backward pass
        self.input_shape = input_tensor.shape
        num_features = input_tensor.shape[-1] # Last dimension is features/channels

        if self.gamma is None: # Initialize parameters on first forward pass
            self.initialize_parameters(num_features)

        if self.training_mode:
            # Calculate mean and variance over the current batch.
            # For Conv layers (B, H, W, C), normalize over B, H, W for each channel C.
            # For Dense layers (B, D), normalize over B for each feature D.
            axes_to_reduce = tuple(range(len(self.input_shape) - 1)) # e.g., (0,1,2) for Conv, (0,) for Dense

            self.batch_mean = np.mean(input_tensor, axis=axes_to_reduce, keepdims=True)
            self.batch_variance = np.var(input_tensor, axis=axes_to_reduce, keepdims=True)
            
            # Normalize the input: x_hat = (x - mu) / sqrt(var + epsilon)
            self.inv_stddev = 1. / np.sqrt(self.batch_variance + self.epsilon)
            self.x_normalized = (input_tensor - self.batch_mean) * self.inv_stddev
            
            # Scale and shift: y = gamma * x_hat + beta
            self.output = self.gamma * self.x_normalized + self.beta

            # Update moving averages for use during inference
            self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * self.batch_mean
            self.moving_variance = self.momentum * self.moving_variance + (1 - self.momentum) * self.batch_variance
        else: # Inference mode: use moving averages
            x_normalized_inference = (input_tensor - self.moving_mean) * (1. / np.sqrt(self.moving_variance + self.epsilon))
            self.output = self.gamma * x_normalized_inference + self.beta
            
        return self.output

    def backward(self, output_gradient):
        # output_gradient (dL/dO) has the same shape as self.output
        
        # Number of elements over which mean/variance were computed (N*H*W for conv, N for dense)
        # This is N' in the BN paper equations (m in some notations)
        N_prime = np.prod(self.input_shape[:-1]) if len(self.input_shape) > 1 else self.input_shape[0]


        # 1. Gradient w.r.t. learnable parameters gamma and beta
        # dL/dGamma = sum(dL/dO * X_normalized) over N, H, W (or N for dense)
        # dL/dBeta = sum(dL/dO) over N, H, W (or N for dense)
        summation_axes = tuple(range(len(self.input_shape) - 1))
        self.d_gamma = np.sum(output_gradient * self.x_normalized, axis=summation_axes, keepdims=True)
        self.d_beta = np.sum(output_gradient, axis=summation_axes, keepdims=True)

        # 2. Gradient w.r.t. normalized input (dL/dX_normalized)
        # dL/dX_norm = dL/dO * gamma
        dx_normalized = output_gradient * self.gamma # Element-wise

        # 3. Gradient w.r.t. variance (dL/dVar)
        # dL/dVar = sum(dL/dX_norm * (X - mu) * (-0.5) * (var + eps)^(-1.5))
        # (X - mu) is self.input_tensor - self.batch_mean
        # (var + eps)^(-0.5) is self.inv_stddev
        # (var + eps)^(-1.5) is self.inv_stddev**3
        dx_minus_mean = self.input_tensor - self.batch_mean
        dvariance = np.sum(dx_normalized * dx_minus_mean * (-0.5) * (self.inv_stddev**3), 
                           axis=summation_axes, keepdims=True)

        # 4. Gradient w.r.t. mean (dL/dMu)
        # dL/dMu = sum(dL/dX_norm * (-1/sqrt(var+eps))) + dL/dVar * (sum(-2*(X-mu))/N_prime)
        dmean_term1 = np.sum(dx_normalized * (-self.inv_stddev), axis=summation_axes, keepdims=True)
        dmean_term2 = dvariance * np.sum(-2. * dx_minus_mean, axis=summation_axes, keepdims=True) / N_prime
        dmean = dmean_term1 + dmean_term2
        
        # 5. Gradient w.r.t. input X (dL/dX)
        # dL/dX = (dL/dX_norm * (1/sqrt(var+eps))) + (dL/dVar * (2*(X-mu)/N_prime)) + (dL/dMu * (1/N_prime))
        # This distributes the gradients dL/dX_norm, dL/dVar, dL/dMu back to each X_i
        
        dx_term1_dist = dx_normalized * self.inv_stddev
        dx_term2_dist = dvariance * (2. * dx_minus_mean / N_prime)
        dx_term3_dist = dmean / N_prime # Broadcasts dmean to all elements contributing to it
        
        d_input = dx_term1_dist + dx_term2_dist + dx_term3_dist
        
        return d_input
