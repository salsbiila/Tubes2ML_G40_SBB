import numpy as np

class ActivationFunction:
    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    @staticmethod
    def softplus(x):
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
    
    @staticmethod
    def elu(x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    @staticmethod
    def selu(x):
        alpha = 1.67326
        scale = 1.0507
        return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    @staticmethod
    def prelu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def swish(x):
        return x * 1 / (1 + np.exp(-np.clip(x, -500, 500)))


class DenseLayer:
    def __init__(self, weights, bias=None, activation='linear', activation_params=None):
        self.weights = weights
        self.bias = bias
        self.activation = activation
        self.activation_params = activation_params or {}
    
    def _apply_activation(self, z):
        if hasattr(ActivationFunction, self.activation):
            activation_func = getattr(ActivationFunction, self.activation)
            return activation_func(z, **self.activation_params)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def forward(self, x):
        z = np.dot(x, self.weights)
        if self.bias is not None:
            z += self.bias

        return self._apply_activation(z)