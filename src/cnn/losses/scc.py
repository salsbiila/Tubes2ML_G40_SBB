import numpy as np

class SparseCategoricalCrossentropy:
    def __init__(self, from_logits=True):
        self.from_logits = from_logits
        self.probabilities = None

    def forward(self, y_pred_logits_or_probs, y_true_sparse):
        """
        y_pred_logits_or_probs: Output from the model (logits or probabilities). Shape (batch_size, num_classes)
        y_true_sparse: True class indices. Shape (batch_size,) or (batch_size, 1)
        """
        batch_size = y_pred_logits_or_probs.shape[0]
        num_classes = y_pred_logits_or_probs.shape[1]
        
        # Ensure y_true_sparse is flat (batch_size,)
        if y_true_sparse.ndim > 1 and y_true_sparse.shape[1] == 1:
            y_true_sparse = y_true_sparse.flatten()

        if self.from_logits:
            # Apply softmax to convert logits to probabilities
            exp_logits = np.exp(y_pred_logits_or_probs - np.max(y_pred_logits_or_probs, axis=1, keepdims=True))
            self.probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        else:
            self.probabilities = y_pred_logits_or_probs # Assume input is already probabilities

        # Clip probabilities to avoid log(0)
        epsilon = 1e-12 # Small constant
        self.probabilities = np.clip(self.probabilities, epsilon, 1. - epsilon)
        
        # Select the predicted probability for the true class for each sample
        # y_true_sparse contains the indices of the true classes
        correct_log_probs = -np.log(self.probabilities[np.arange(batch_size), y_true_sparse])
        
        # Compute the mean loss over the batch
        loss = np.mean(correct_log_probs)
        return loss

    def backward(self, y_pred_logits_or_probs, y_true_sparse):
        """
        Computes the gradient of the loss w.r.t. the input of the loss function.
        If from_logits=True, this is dL/d(logits).
        If from_logits=False, this is dL/d(probabilities) (more complex, not fully handled here).
        
        y_true_sparse: True class indices. Shape (batch_size,) or (batch_size, 1)
        """
        batch_size = y_pred_logits_or_probs.shape[0]
        num_classes = y_pred_logits_or_probs.shape[1]

        if y_true_sparse.ndim > 1 and y_true_sparse.shape[1] == 1:
            y_true_sparse = y_true_sparse.flatten()

        # The gradient of SparseCategoricalCrossentropy + Softmax w.r.t logits is (P - Y_one_hot)
        # where P is the softmax output and Y_one_hot is the one-hot encoded true labels.
        
        # Probabilities (P) were calculated in forward pass (or are the input if not from_logits)
        # If self.probabilities is not set (e.g., backward called before forward), recalculate
        if self.probabilities is None or self.probabilities.shape != y_pred_logits_or_probs.shape:
             if self.from_logits:
                exp_logits = np.exp(y_pred_logits_or_probs - np.max(y_pred_logits_or_probs, axis=1, keepdims=True))
                self.probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
             else:
                self.probabilities = y_pred_logits_or_probs


        gradient = np.copy(self.probabilities)
        
        # Subtract 1 from the probability of the true class for each sample (P_i - 1 for true class i)
        gradient[np.arange(batch_size), y_true_sparse] -= 1
        
        # Normalize by batch size
        gradient = gradient / batch_size
        
        # This gradient (dL/d(logits)) is what the layer *before* softmax expects.
        # If the network's last layer IS Softmax, and from_logits=True was used,
        # then this gradient should be passed to the layer that produced the logits.
        # If from_logits=False, the gradient calculation dL/d(Probabilities) is more complex.
        # For simplicity and common usage, we'll assume from_logits=True.
        if not self.from_logits:
            # This case is more complex. dL/dp_i = - y_i / p_i for the correct class, 0 otherwise.
            # Then this needs to be backpropagated through the softmax layer itself if it exists.
            # For simplicity, this scratch version is best used with from_logits=True.
            raise NotImplementedError("Backward pass for from_logits=False with SCCE is not fully implemented here. Use from_logits=True.")
            
        return gradient