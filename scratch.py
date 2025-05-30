import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
from sklearn.metrics import f1_score, accuracy_score

def _to_numpy(x):
    """Convert TensorFlow tensor or other array-like to NumPy array"""
    if hasattr(x, 'numpy'):
        return x.numpy()
    return np.array(x)

def _safe_copy(x):
    """Safely copy array (handle both NumPy and TensorFlow tensors)"""
    if hasattr(x, 'numpy'):
        return x.numpy().copy()
    return np.array(x).copy()

class EmbeddingLayer:
    def __init__(self, weights):
        self.weights = _to_numpy(weights).copy()
        self.vocab_size, self.embedding_dim = weights.shape
        
        # For backward pass
        self.last_input = None
        self.gradients = {'weights': np.zeros_like(self.weights)}
    
    def forward(self, x):
        # Store input for backward pass
        self.last_input = _safe_copy(x)
        x_np = _to_numpy(x)
        return self.weights[x_np]
    
    def backward(self, grad_output):
        """
        grad_output: (batch_size, seq_len, embedding_dim) or (batch_size, embedding_dim)
        """
        if grad_output is None:
            return None
            
        # Convert to numpy
        grad_output = _to_numpy(grad_output)
        
        # Reset gradients
        self.gradients['weights'] = np.zeros_like(self.weights)
        
        # Handle different input shapes
        if len(grad_output.shape) == 3:
            batch_size, seq_len, embedding_dim = grad_output.shape
            # Flatten for easier processing
            grad_flat = grad_output.reshape(-1, embedding_dim)
            input_flat = self.last_input.reshape(-1)
        else:
            grad_flat = grad_output
            input_flat = self.last_input.reshape(-1)
        
        # Accumulate gradients for each unique token
        for i, token_id in enumerate(input_flat):
            if 0 <= token_id < self.vocab_size:
                self.gradients['weights'][token_id] += grad_flat[i]
        
        # No gradient w.r.t input (tokens are discrete)
        return None
    
    def update_weights(self, learning_rate):
        self.weights -= learning_rate * self.gradients['weights']

class SimpleRNNCell:
    def __init__(self, kernel, recurrent_kernel, bias=None, use_bias=True):
        self.kernel = _to_numpy(kernel).copy()
        self.recurrent_kernel = _to_numpy(recurrent_kernel).copy()
        self.bias = _to_numpy(bias).copy() if bias is not None else None
        self.use_bias = use_bias
        self.units = recurrent_kernel.shape[0]
        
        # For backward pass
        self.gradients = {
            'kernel': np.zeros_like(self.kernel),
            'recurrent_kernel': np.zeros_like(self.recurrent_kernel),
            'bias': np.zeros_like(self.bias) if self.bias is not None else None
        }
    
    def forward_step(self, x_t, h_prev):
        # Convert inputs to numpy
        x_t = _to_numpy(x_t)
        h_prev = _to_numpy(h_prev)
        
        # Store values for backward pass
        self.last_x_t = x_t.copy()
        self.last_h_prev = h_prev.copy()
        
        # Forward computation
        linear = np.dot(x_t, self.kernel) + np.dot(h_prev, self.recurrent_kernel)
        if self.use_bias and self.bias is not None:
            linear += self.bias
        
        self.last_linear = linear.copy()
        h_t = np.tanh(linear)
        self.last_h_t = h_t.copy()
        
        return h_t
    
    def backward_step(self, grad_h_t):
        """
        Backward pass for single time step
        grad_h_t: gradient w.r.t hidden state at time t
        Returns: grad_x_t, grad_h_prev
        """
        # Convert to numpy
        grad_h_t = _to_numpy(grad_h_t)
        
        # Gradient through tanh activation
        # tanh'(x) = 1 - tanh²(x)
        grad_linear = grad_h_t * (1 - self.last_h_t ** 2)
        
        # Gradients w.r.t weights
        self.gradients['kernel'] += np.dot(self.last_x_t.T, grad_linear)
        self.gradients['recurrent_kernel'] += np.dot(self.last_h_prev.T, grad_linear)
        
        if self.use_bias and self.bias is not None:
            self.gradients['bias'] += np.sum(grad_linear, axis=0)
        
        # Gradients w.r.t inputs
        grad_x_t = np.dot(grad_linear, self.kernel.T)
        grad_h_prev = np.dot(grad_linear, self.recurrent_kernel.T)
        
        return grad_x_t, grad_h_prev
    
    def reset_gradients(self):
        self.gradients['kernel'] = np.zeros_like(self.kernel)
        self.gradients['recurrent_kernel'] = np.zeros_like(self.recurrent_kernel)
        if self.gradients['bias'] is not None:
            self.gradients['bias'] = np.zeros_like(self.bias)
    
    def update_weights(self, learning_rate):
        self.kernel -= learning_rate * self.gradients['kernel']
        self.recurrent_kernel -= learning_rate * self.gradients['recurrent_kernel']
        if self.bias is not None:
            self.bias -= learning_rate * self.gradients['bias']

class SimpleRNNLayer:
    def __init__(self, weights, return_sequences=False, go_backwards=False):
        # Initialize RNN cell
        if len(weights) == 3:
            kernel, recurrent_kernel, bias = weights
            self.cell = SimpleRNNCell(kernel, recurrent_kernel, bias, use_bias=True)
        elif len(weights) == 2:
            kernel, recurrent_kernel = weights
            self.cell = SimpleRNNCell(kernel, recurrent_kernel, None, use_bias=False)
        else:
            raise ValueError(f"Expected 2 or 3 weight arrays, got {len(weights)}")
        
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards
        
        # For backward pass
        self.last_input = None
        self.hidden_states_history = []
    
    def forward(self, x):
        # Convert to numpy
        x = _to_numpy(x)
        batch_size, sequence_length, input_dim = x.shape
        
        # Store input for backward pass
        self.last_input = x.copy()
        
        # Reset hidden states history
        self.hidden_states_history = []
        
        # If go_backwards, reverse the sequence
        if self.go_backwards:
            x = x[:, ::-1, :]
        
        hidden_states = []
        h_t = np.zeros((batch_size, self.cell.units))
        
        # Store initial hidden state
        self.hidden_states_history.append(h_t.copy())
        
        # Process each time step
        for t in range(sequence_length):
            x_t = x[:, t, :]
            h_t = self.cell.forward_step(x_t, h_t)
            hidden_states.append(h_t.copy())
            self.hidden_states_history.append(h_t.copy())
        
        if self.return_sequences:
            output = np.stack(hidden_states, axis=1)
            if self.go_backwards:
                output = output[:, ::-1, :]
            return output
        else:
            return hidden_states[-1]
    
    def backward(self, grad_output):
        """
        Backpropagation Through Time (BPTT)
        """
        if self.last_input is None:
            raise ValueError("Must call forward() before backward()")
        
        if grad_output is None:
            return None
            
        # Convert to numpy
        grad_output = _to_numpy(grad_output)
        
        batch_size, sequence_length, input_dim = self.last_input.shape
        
        # Reset cell gradients
        self.cell.reset_gradients()
        
        # Handle different output shapes
        if self.return_sequences:
            if self.go_backwards:
                grad_output = grad_output[:, ::-1, :]
            grad_h_sequence = grad_output
        else:
            # Create gradient for all time steps, but only last step has gradient
            grad_h_sequence = np.zeros((batch_size, sequence_length, self.cell.units))
            grad_h_sequence[:, -1, :] = grad_output
        
        # Initialize gradients
        grad_input = np.zeros_like(self.last_input)
        grad_h_next = np.zeros((batch_size, self.cell.units))
        
        # Backpropagate through time (reverse order)
        input_sequence = self.last_input
        if self.go_backwards:
            input_sequence = input_sequence[:, ::-1, :]
        
        for t in reversed(range(sequence_length)):
            # Get stored values for this timestep
            x_t = input_sequence[:, t, :]
            h_prev = self.hidden_states_history[t]
            h_t = self.hidden_states_history[t + 1]
            
            # Store in cell for backward step
            self.cell.last_x_t = x_t
            self.cell.last_h_prev = h_prev
            self.cell.last_h_t = h_t
            self.cell.last_linear = np.dot(x_t, self.cell.kernel) + np.dot(h_prev, self.cell.recurrent_kernel)
            if self.cell.use_bias and self.cell.bias is not None:
                self.cell.last_linear += self.cell.bias
            
            # Total gradient for this timestep
            total_grad_h = grad_h_sequence[:, t, :] + grad_h_next
            
            # Backward step
            grad_x_t, grad_h_prev = self.cell.backward_step(total_grad_h)
            
            # Store input gradient
            if self.go_backwards:
                grad_input[:, sequence_length - 1 - t, :] = grad_x_t
            else:
                grad_input[:, t, :] = grad_x_t
            
            # Update gradient for next iteration
            grad_h_next = grad_h_prev
        
        return grad_input
    
    def update_weights(self, learning_rate):
        self.cell.update_weights(learning_rate)

class BidirectionalRNNLayer:
    def __init__(self, forward_weights, backward_weights, merge_mode='concat'):
        self.forward_rnn = SimpleRNNLayer(forward_weights, return_sequences=True, go_backwards=False)
        self.backward_rnn = SimpleRNNLayer(backward_weights, return_sequences=True, go_backwards=True)
        self.merge_mode = merge_mode
        
        # For backward pass
        self.last_forward_output = None
        self.last_backward_output = None
    
    def forward(self, x):
        # Forward pass
        self.last_forward_output = self.forward_rnn.forward(x)
        
        # Backward pass
        self.last_backward_output = self.backward_rnn.forward(x)
        
        # For final output, we want the last timestep from each direction
        forward_last = self.last_forward_output[:, -1, :]
        backward_last = self.last_backward_output[:, 0, :]
        
        if self.merge_mode == 'concat':
            output = np.concatenate([forward_last, backward_last], axis=1)
        elif self.merge_mode == 'sum':
            output = forward_last + backward_last
        elif self.merge_mode == 'avg':
            output = (forward_last + backward_last) / 2
        else:
            raise ValueError(f"Unsupported merge_mode: {self.merge_mode}")
        
        return output
    
    def backward(self, grad_output):
        if grad_output is None:
            return None
            
        # Convert to numpy
        grad_output = _to_numpy(grad_output)
        batch_size = grad_output.shape[0]
        
        if self.merge_mode == 'concat':
            # Split gradient
            units = grad_output.shape[1] // 2
            grad_forward_last = grad_output[:, :units]
            grad_backward_last = grad_output[:, units:]
        elif self.merge_mode == 'sum':
            grad_forward_last = grad_output
            grad_backward_last = grad_output
        elif self.merge_mode == 'avg':
            grad_forward_last = grad_output / 2
            grad_backward_last = grad_output / 2
        
        # Create gradients for full sequences
        seq_len = self.last_forward_output.shape[1]
        
        # Forward RNN: gradient only at last timestep
        grad_forward_seq = np.zeros_like(self.last_forward_output)
        grad_forward_seq[:, -1, :] = grad_forward_last
        
        # Backward RNN: gradient only at first timestep (which was last in processing)
        grad_backward_seq = np.zeros_like(self.last_backward_output)
        grad_backward_seq[:, 0, :] = grad_backward_last
        
        # Backward pass through each RNN
        grad_input_forward = self.forward_rnn.backward(grad_forward_seq)
        grad_input_backward = self.backward_rnn.backward(grad_backward_seq)
        
        # Handle None gradients
        if grad_input_forward is None and grad_input_backward is None:
            return None
        elif grad_input_forward is None:
            return grad_input_backward
        elif grad_input_backward is None:
            return grad_input_forward
        else:
            # Combine input gradients
            return grad_input_forward + grad_input_backward
    
    def update_weights(self, learning_rate):
        self.forward_rnn.update_weights(learning_rate)
        self.backward_rnn.update_weights(learning_rate)

class DenseLayer:
    def __init__(self, weights, bias=None, activation='linear'):
        self.weights = _to_numpy(weights).copy()
        self.bias = _to_numpy(bias).copy() if bias is not None else None
        self.activation = activation
        
        # For backward pass
        self.last_input = None
        self.last_linear_output = None
        self.last_output = None
        self.gradients = {
            'weights': np.zeros_like(self.weights),
            'bias': np.zeros_like(self.bias) if self.bias is not None else None
        }
    
    def _apply_activation(self, z):
        if self.activation == 'linear':
            return z
        elif self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation == 'softmax':
            z_shifted = z - np.max(z, axis=1, keepdims=True)
            exp_z = np.exp(z_shifted)
            return exp_z / np.sum(exp_z, axis=1, keepdims=True)
        elif self.activation == 'tanh':
            return np.tanh(z)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def _activation_derivative(self, z, output):
        if self.activation == 'linear':
            return np.ones_like(z)
        elif self.activation == 'relu':
            return (z > 0).astype(float)
        elif self.activation == 'sigmoid':
            return output * (1 - output)
        elif self.activation == 'softmax':
            # For softmax, derivative is handled differently in loss
            return np.ones_like(z)
        elif self.activation == 'tanh':
            return 1 - output ** 2
    
    def forward(self, x):
        # Convert to numpy and store input for backward pass
        x = _to_numpy(x)
        self.last_input = x.copy()
        
        # Linear transformation
        z = np.dot(x, self.weights)
        if self.bias is not None:
            z += self.bias
        
        self.last_linear_output = z.copy()
        
        # Apply activation
        output = self._apply_activation(z)
        self.last_output = output.copy()
        
        return output
    
    def backward(self, grad_output):
        if self.last_input is None:
            raise ValueError("Must call forward() before backward()")
        
        if grad_output is None:
            return None
            
        # Convert to numpy
        grad_output = _to_numpy(grad_output)
        
        # Gradient through activation
        if self.activation == 'softmax':
            # For softmax with cross-entropy, gradient is simplified
            grad_linear = grad_output
        else:
            activation_grad = self._activation_derivative(self.last_linear_output, self.last_output)
            grad_linear = grad_output * activation_grad
        
        # Gradients w.r.t weights and bias
        self.gradients['weights'] = np.dot(self.last_input.T, grad_linear)
        if self.bias is not None:
            self.gradients['bias'] = np.sum(grad_linear, axis=0)
        
        # Gradient w.r.t input
        grad_input = np.dot(grad_linear, self.weights.T)
        
        return grad_input
    
    def update_weights(self, learning_rate):
        self.weights -= learning_rate * self.gradients['weights']
        if self.bias is not None:
            self.bias -= learning_rate * self.gradients['bias']

class SimpleRNNModelFromScratch:
    def __init__(self, keras_model_path):
        self.model = keras.models.load_model(keras_model_path)
        self.layers = []
        self._build_layers()
    
    def _get_activation_name(self, activation_func):
        if hasattr(activation_func, '__name__'):
            return activation_func.__name__
        elif hasattr(activation_func, 'name'):
            return activation_func.name
        else:
            return str(activation_func).split('.')[-1].split(' ')[0]
    
    def _build_layers(self):
        print("Building layers from Keras model...")
        print(f"Model has {len(self.model.layers)} layers")
        
        for i, layer in enumerate(self.model.layers):
            layer_type = layer.__class__.__name__
            print(f"Layer {i}: {layer_type} - {layer.name}")
            
            if layer_type == 'Embedding':
                weights = layer.get_weights()
                if len(weights) > 0:
                    embedding_weights = weights[0]
                    print(f"  Embedding weights shape: {embedding_weights.shape}")
                    self.layers.append(EmbeddingLayer(embedding_weights))
            
            elif layer_type == 'Bidirectional':
                print(f"  Processing Bidirectional layer...")
                all_weights = layer.get_weights()
                print(f"  Total weights in bidirectional layer: {len(all_weights)}")
                
                if len(all_weights) == 6:
                    forward_weights = all_weights[:3]
                    backward_weights = all_weights[3:6]
                elif len(all_weights) == 4:
                    forward_weights = all_weights[:2]
                    backward_weights = all_weights[2:4]
                else:
                    raise ValueError(f"Unexpected number of weights: {len(all_weights)}")
                
                merge_mode = getattr(layer, 'merge_mode', 'concat')
                self.layers.append(BidirectionalRNNLayer(forward_weights, backward_weights, merge_mode))
            
            elif layer_type == 'SimpleRNN':
                weights = layer.get_weights()
                return_seq = getattr(layer, 'return_sequences', False)
                go_back = getattr(layer, 'go_backwards', False)
                self.layers.append(SimpleRNNLayer(weights, return_sequences=return_seq, go_backwards=go_back))
            
            elif layer_type == 'Dense':
                weights = layer.get_weights()
                if len(weights) == 0:
                    continue
                
                dense_weights = weights[0]
                dense_bias = weights[1] if len(weights) > 1 else None
                activation = self._get_activation_name(layer.activation)
                
                self.layers.append(DenseLayer(dense_weights, dense_bias, activation=activation))
            
            elif layer_type == 'Dropout':
                continue
        
        print(f"Built {len(self.layers)} functional layers")
    
    def forward(self, x):
        # Convert input to numpy if needed
        current_output = _to_numpy(x)
        for layer in self.layers:
            current_output = layer.forward(current_output)
        return current_output
    
    def backward(self, y_true, y_pred):
        """
        Backward pass through entire model
        y_true: true labels (integers)
        y_pred: predicted probabilities
        """
        # Convert to numpy
        y_true = _to_numpy(y_true)
        y_pred = _to_numpy(y_pred)
        
        batch_size = y_pred.shape[0]
        num_classes = y_pred.shape[1]
        
        # Convert labels to one-hot for gradient computation
        y_true_onehot = np.zeros_like(y_pred)
        y_true_onehot[np.arange(batch_size), y_true] = 1
        
        # Gradient of cross-entropy loss w.r.t output
        # For softmax + cross-entropy: grad = y_pred - y_true
        grad_output = y_pred - y_true_onehot
        
        # Backward pass through all layers
        current_grad = grad_output
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                current_grad = layer.backward(current_grad)
                # Handle None gradients (e.g., from Embedding layer)
                if current_grad is None:
                    break
    
    def update_weights(self, learning_rate):
        """Update weights for all layers"""
        for layer in self.layers:
            if hasattr(layer, 'update_weights'):
                layer.update_weights(learning_rate)
    
    def compute_loss(self, y_true, y_pred):
        """Compute cross-entropy loss"""
        # Convert to numpy
        y_true = _to_numpy(y_true)
        y_pred = _to_numpy(y_pred)
        
        # Avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        # Cross-entropy loss
        batch_size = y_true.shape[0]
        loss = -np.sum(np.log(y_pred_clipped[np.arange(batch_size), y_true])) / batch_size
        
        return loss
    
    def train_step(self, x, y_true, learning_rate=0.001):
        """Single training step"""
        # Convert to numpy
        x = _to_numpy(x)
        y_true = _to_numpy(y_true)
        
        # Forward pass
        y_pred = self.forward(x)
        
        # Compute loss
        loss = self.compute_loss(y_true, y_pred)
        
        # Backward pass
        self.backward(y_true, y_pred)
        
        # Update weights
        self.update_weights(learning_rate)
        
        return loss
    
    def train(self, x_train, y_train, x_val=None, y_val=None, epochs=10, batch_size=32, learning_rate=0.001):
        """Training loop"""
        # Convert to numpy
        x_train = _to_numpy(x_train)
        y_train = _to_numpy(y_train)
        if x_val is not None:
            x_val = _to_numpy(x_val)
        if y_val is not None:
            y_val = _to_numpy(y_val)
            
        train_losses = []
        val_losses = []
        
        n_samples = len(x_train)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            x_train_shuffled = x_train[indices]
            y_train_shuffled = y_train[indices]
            
            epoch_losses = []
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                x_batch = x_train_shuffled[i:end_idx]
                y_batch = y_train_shuffled[i:end_idx]
                
                loss = self.train_step(x_batch, y_batch, learning_rate)
                epoch_losses.append(loss)
            
            avg_train_loss = np.mean(epoch_losses)
            train_losses.append(avg_train_loss)
            
            # Validation
            if x_val is not None and y_val is not None:
                val_pred = self.forward(x_val[:100])  # Use subset for validation
                val_loss = self.compute_loss(y_val[:100], val_pred)
                val_losses.append(val_loss)
                
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f}")
        
        return train_losses, val_losses
    
    def predict(self, x):
        probabilities = self.forward(x)
        return np.argmax(probabilities, axis=1)

# Test functions remain the same
def debug_model_structure(keras_model_path):
    """Debug function to understand model structure"""
    print("=== MODEL STRUCTURE DEBUG ===")
    
    model = keras.models.load_model(keras_model_path)
    print(f"Model type: {type(model)}")
    print(f"Number of layers: {len(model.layers)}")
    print()
    
    for i, layer in enumerate(model.layers):
        print(f"Layer {i}: {layer.__class__.__name__}")
        print(f"  Name: {layer.name}")
        
        weights = layer.get_weights()
        print(f"  Weights count: {len(weights)}")
        for j, w in enumerate(weights):
            print(f"    Weight {j} shape: {w.shape}")
        
        if hasattr(layer, 'activation'):
            print(f"  Activation: {layer.activation}")
        if hasattr(layer, 'return_sequences'):
            print(f"  Return sequences: {layer.return_sequences}")
        
        print()

def compare_implementations(keras_model_path, test_data, test_labels, max_samples=100):
    """Compare Keras and from-scratch implementations"""
    print("=== IMPLEMENTATION COMPARISON ===")
    
    # Convert to NumPy arrays
    test_data = _to_numpy(test_data)
    test_labels = _to_numpy(test_labels)
    
    # Use subset for testing
    if len(test_data) > max_samples:
        indices = np.random.choice(len(test_data), max_samples, replace=False)
        test_data_subset = test_data[indices]
        test_labels_subset = test_labels[indices]
    else:
        test_data_subset = test_data
        test_labels_subset = test_labels
    
    print(f"Using {len(test_data_subset)} samples for comparison")
    
    # Load models
    keras_model = keras.models.load_model(keras_model_path)
    scratch_model = SimpleRNNModelFromScratch(keras_model_path)
    
    # Get predictions
    print("\nGetting Keras predictions...")
    keras_predictions = keras_model.predict(test_data_subset, verbose=0)
    keras_classes = np.argmax(keras_predictions, axis=1)
    
    print("\nGetting from-scratch predictions...")
    scratch_predictions = scratch_model.forward(test_data_subset)
    scratch_classes = np.argmax(scratch_predictions, axis=1)
    
    # Calculate metrics
    keras_f1 = f1_score(test_labels_subset, keras_classes, average='macro')
    scratch_f1 = f1_score(test_labels_subset, scratch_classes, average='macro')
    
    keras_acc = accuracy_score(test_labels_subset, keras_classes)
    scratch_acc = accuracy_score(test_labels_subset, scratch_classes)
    
    prediction_agreement = np.mean(keras_classes == scratch_classes)
    max_diff = np.max(np.abs(keras_predictions - scratch_predictions))
    mean_diff = np.mean(np.abs(keras_predictions - scratch_predictions))
    
    print(f"\n=== RESULTS ===")
    print(f"Keras Model:")
    print(f"  Accuracy: {keras_acc:.4f}")
    print(f"  Macro F1: {keras_f1:.4f}")
    print(f"\nFrom-Scratch Model:")
    print(f"  Accuracy: {scratch_acc:.4f}")
    print(f"  Macro F1: {scratch_f1:.4f}")
    print(f"\nPrediction Agreement: {prediction_agreement:.4f}")
    print(f"Max Absolute Difference: {max_diff:.6f}")
    print(f"Mean Absolute Difference: {mean_diff:.6f}")
    
    return {
        'keras_f1': keras_f1,
        'scratch_f1': scratch_f1,
        'prediction_agreement': prediction_agreement,
        'max_diff': max_diff,
        'mean_diff': mean_diff
    }