import numpy as np

class SparseCategoricalCrossentropy:
    def __init__(self, from_logits=True):
        self.from_logits = from_logits
        self.probabilities = None

    def forward(self, y_pred_logits_or_probs, y_true_sparse):
        batch_size = y_pred_logits_or_probs.shape[0]
        
        if y_true_sparse.ndim > 1 and y_true_sparse.shape[1] == 1:
            y_true_sparse = y_true_sparse.flatten()

        if self.from_logits:
            exp_logits = np.exp(y_pred_logits_or_probs - np.max(y_pred_logits_or_probs, axis=1, keepdims=True))
            self.probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        else:
            self.probabilities = y_pred_logits_or_probs

        epsilon = 1e-12
        self.probabilities = np.clip(self.probabilities, epsilon, 1. - epsilon)
        
        correct_log_probs = -np.log(self.probabilities[np.arange(batch_size), y_true_sparse])
        
        loss = np.mean(correct_log_probs)
        return loss

    def backward(self, y_pred_logits_or_probs, y_true_sparse):
        batch_size = y_pred_logits_or_probs.shape[0]

        if y_true_sparse.ndim > 1 and y_true_sparse.shape[1] == 1:
            y_true_sparse = y_true_sparse.flatten()

        if self.probabilities is None or self.probabilities.shape != y_pred_logits_or_probs.shape:
             if self.from_logits:
                exp_logits = np.exp(y_pred_logits_or_probs - np.max(y_pred_logits_or_probs, axis=1, keepdims=True))
                self.probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
             else:
                self.probabilities = y_pred_logits_or_probs


        gradient = np.copy(self.probabilities)
        
        gradient[np.arange(batch_size), y_true_sparse] -= 1
        
        gradient = gradient / batch_size
            
        return gradient