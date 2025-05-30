# src/cnn/cnn.py
import numpy as np
import tensorflow as tf 
from .layers import Conv2DLayer, ReLULayer, PoolingLayer, FlattenLayer, DenseLayer, SoftmaxLayer

class CNN:
    def __init__(self):
        self.layers = []
        self.keras_optimizer = None
        self.loss_function = None

    def add_layer(self, layer):
        self.layers.append(layer)

    def compile_model(self, optimizer, loss_function):
        if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
            raise ValueError("Optimizer must be an instance of tf.keras.optimizers.Optimizer")
        self.keras_optimizer = optimizer
        self.loss_function = loss_function

    def _prepare_input(self, input_tensor):
        original_ndim = input_tensor.ndim
        x = input_tensor
        was_single = False

        expected_single_ndim = -1
        if self.layers:
            first_layer = self.layers[0]
            if isinstance(first_layer, (Conv2DLayer, PoolingLayer)):
                expected_single_ndim = 3
            elif isinstance(first_layer, (DenseLayer, FlattenLayer, ReLULayer)): 
                if isinstance(first_layer, DenseLayer):
                    expected_single_ndim = 1
                elif len(input_tensor.shape) > 0 and input_tensor.shape[-1] > 3 :
                    expected_single_ndim = 1


        if expected_single_ndim != -1 and original_ndim == expected_single_ndim:
            x = np.expand_dims(input_tensor, axis=0)
            was_single = True
        elif original_ndim == expected_single_ndim + 1:
            pass 
        else:
            pass
            
        return x, was_single

    def _unprepare_output(self, output_tensor, was_single_instance):
        if was_single_instance and output_tensor.shape[0] == 1:
            return np.squeeze(output_tensor, axis=0)
        return output_tensor

    def forward(self, input_tensor, training=True):
        x, _ = self._prepare_input(input_tensor)
        
        for layer in self.layers:
            layer.set_training_mode(training)
            x = layer.forward(x)
            
        return x

    def backward(self, loss_gradient):
        grad = loss_gradient
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def fit(self, X_train, y_train, epochs, batch_size, X_val=None, y_val=None):
        if self.keras_optimizer is None or self.loss_function is None:
            raise ValueError("Model not compiled. Call compile_model with a Keras optimizer and a loss function.")

        num_samples = X_train.shape[0]
        
        if batch_size is None or not isinstance(batch_size, int) or batch_size <= 0:
            print(f"Info: batch_size is '{batch_size}'. Using full batch (batch_size = {num_samples}).")
            batch_size = num_samples
        elif batch_size > num_samples:
            print(f"Warning: batch_size ({batch_size}) > num_samples ({num_samples}). Setting batch_size to {num_samples}.")
            batch_size = num_samples


        history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_correct_preds = 0
            
            permutation = np.random.permutation(num_samples)
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]

            for i in range(0, num_samples, batch_size):
                X_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]

                y_pred_output_batched = self.forward(X_batch, training=True)

                loss = self.loss_function.forward(y_pred_output_batched, y_batch)
                epoch_loss += loss * X_batch.shape[0]

                loss_gradient_batched = self.loss_function.backward(y_pred_output_batched, y_batch)

                self.backward(loss_gradient_batched)

                trainable_params_numpy = []
                gradients_numpy = []
                for layer in self.layers:
                    if hasattr(layer, 'weights') and layer.weights is not None and hasattr(layer, 'd_weights') and layer.d_weights is not None:
                        trainable_params_numpy.append(layer.weights)
                        gradients_numpy.append(layer.d_weights)

                    if hasattr(layer, 'biases') and layer.biases is not None and hasattr(layer, 'd_biases') and layer.d_biases is not None:
                        trainable_params_numpy.append(layer.biases)
                        gradients_numpy.append(layer.d_biases)
                
                if trainable_params_numpy: 
                    trainable_params_tf = [tf.Variable(p) for p in trainable_params_numpy]
                    gradients_tf = [tf.convert_to_tensor(g, dtype=tf.float32) for g in gradients_numpy]

                    self.keras_optimizer.apply_gradients(zip(gradients_tf, trainable_params_tf))

                    param_idx = 0
                    for layer in self.layers:
                        if hasattr(layer, 'weights') and layer.weights is not None and \
                           hasattr(layer, 'd_weights') and layer.d_weights is not None:
                            layer.weights = trainable_params_tf[param_idx].numpy()
                            param_idx += 1
                        if hasattr(layer, 'biases') and layer.biases is not None and \
                           hasattr(layer, 'd_biases') and layer.d_biases is not None:
                            layer.biases = trainable_params_tf[param_idx].numpy()
                            param_idx += 1
                
                if self.loss_function.from_logits:
                    exp_logits = np.exp(y_pred_output_batched - np.max(y_pred_output_batched, axis=1, keepdims=True))
                    batch_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
                    batch_preds = np.argmax(batch_probs, axis=1)
                else: 
                    batch_preds = np.argmax(y_pred_output_batched, axis=1)
                epoch_correct_preds += np.sum(batch_preds == y_batch.flatten())

            avg_epoch_loss = epoch_loss / num_samples
            epoch_accuracy = epoch_correct_preds / num_samples
            history['loss'].append(avg_epoch_loss)
            history['accuracy'].append(epoch_accuracy)

            val_info = ""
            if X_val is not None and y_val is not None:
                y_val_output_batched = self.forward(X_val, training=False)
                val_loss = self.loss_function.forward(y_val_output_batched, y_val)
                
                val_probs_for_acc = self.predict_proba(X_val)
                
                if val_probs_for_acc.ndim == 1:
                    val_preds = np.argmax(val_probs_for_acc)
                else: 
                    val_preds = np.argmax(val_probs_for_acc, axis=1)

                val_accuracy = np.mean(val_preds == y_val.flatten())
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)
                val_info = f", val_loss: {val_loss:.4f}, val_accuracy: {val_accuracy:.4f}"

            print(f"Epoch {epoch+1}/{epochs} - loss: {avg_epoch_loss:.4f}, accuracy: {epoch_accuracy:.4f}{val_info}")
        return history

    def predict_proba(self, input_tensor):
        x_batched, was_single = self._prepare_input(input_tensor)
        
        output_batched = self.forward(x_batched, training=False)
        
        probabilities_batched = output_batched
        if not isinstance(self.layers[-1], SoftmaxLayer) and (self.loss_function is None or self.loss_function.from_logits):
            exp_logits = np.exp(output_batched - np.max(output_batched, axis=-1, keepdims=True))
            probabilities_batched = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        return self._unprepare_output(probabilities_batched, was_single)

    def predict(self, input_tensor):
        probabilities = self.predict_proba(input_tensor)
        
        if probabilities.ndim == 1:
            return np.argmax(probabilities)
        else:
            return np.argmax(probabilities, axis=1)

    def load_weights_from_keras(self, keras_model):
        manual_layer_idx = 0
        keras_layer_idx = 0

        print("Attempting to load weights from Keras model...")

        while manual_layer_idx < len(self.layers) and keras_layer_idx < len(keras_model.layers):
            current_manual_layer = self.layers[manual_layer_idx]
            keras_layer = keras_model.layers[keras_layer_idx]
            
            print(f"Comparing Manual: {type(current_manual_layer).__name__} with Keras: {type(keras_layer).__name__}")

            loaded_this_pair = False
            advanced_manual = False

            if isinstance(keras_layer, tf.keras.layers.Conv2D) and isinstance(current_manual_layer, Conv2DLayer):
                weights, biases = keras_layer.get_weights()
                if current_manual_layer.weights is None: 
                    current_manual_layer.initialize_parameters(weights.shape[2])
                current_manual_layer.weights = weights
                current_manual_layer.biases = biases.reshape(1, 1, 1, -1)
                current_manual_layer._input_channels = weights.shape[2]
                current_manual_layer.f_h, current_manual_layer.f_w = weights.shape[0], weights.shape[1]
                current_manual_layer.num_filters = weights.shape[3]
                loaded_this_pair = True; advanced_manual = True

            elif isinstance(keras_layer, tf.keras.layers.Dense) and isinstance(current_manual_layer, DenseLayer):
                weights, biases = keras_layer.get_weights()
                if current_manual_layer.weights is None:
                    current_manual_layer.initialize_parameters(weights.shape[0])
                current_manual_layer.weights = weights
                current_manual_layer.biases = biases.reshape(1, -1)
                current_manual_layer.input_dim = weights.shape[0]
                current_manual_layer.output_dim = weights.shape[1]
                loaded_this_pair = True; advanced_manual = True

            elif isinstance(current_manual_layer, ReLULayer):
                is_keras_direct_relu = isinstance(keras_layer, tf.keras.layers.ReLU)
                is_keras_activation_relu = (
                    isinstance(keras_layer, tf.keras.layers.Activation) and
                    hasattr(keras_layer, 'activation') and
                    getattr(keras_layer.activation, '__name__', None) == 'relu'
                )
                if is_keras_direct_relu or is_keras_activation_relu:
                    print(f"  Matched Keras ReLU type for manual ReLULayer.")
                    loaded_this_pair = True
                    advanced_manual = True

            elif isinstance(keras_layer, tf.keras.layers.MaxPooling2D) and isinstance(current_manual_layer, PoolingLayer) and current_manual_layer.mode == 'max':
                current_manual_layer.pool_h, current_manual_layer.pool_w = keras_layer.pool_size
                current_manual_layer.stride_h, current_manual_layer.stride_w = keras_layer.strides if isinstance(keras_layer.strides, tuple) else (keras_layer.strides, keras_layer.strides)
                loaded_this_pair = True; advanced_manual = True

            elif isinstance(keras_layer, tf.keras.layers.AveragePooling2D) and isinstance(current_manual_layer, PoolingLayer) and current_manual_layer.mode == 'average':
                current_manual_layer.pool_h, current_manual_layer.pool_w = keras_layer.pool_size
                current_manual_layer.stride_h, current_manual_layer.stride_w = keras_layer.strides if isinstance(keras_layer.strides, tuple) else (keras_layer.strides, keras_layer.strides)
                loaded_this_pair = True; advanced_manual = True

            elif isinstance(keras_layer, tf.keras.layers.Flatten) and isinstance(current_manual_layer, FlattenLayer):
                loaded_this_pair = True; advanced_manual = True

            elif isinstance(keras_layer, tf.keras.layers.Activation) and hasattr(keras_layer, 'activation') and keras_layer.activation.__name__ == 'softmax' and isinstance(current_manual_layer, SoftmaxLayer):
                loaded_this_pair = True; advanced_manual = True
            
            if loaded_this_pair:
                print(f"  Matched and processed manual: {type(current_manual_layer).__name__} with Keras: {type(keras_layer).__name__}")
                pass
            elif isinstance(keras_layer, (tf.keras.layers.InputLayer)):
                print(f"  Skipping Keras InputLayer: {keras_layer.name}")
                pass 
            else:
                print(f"  No direct match or load for Keras: {type(keras_layer).__name__} with manual: {type(current_manual_layer).__name__}. Advancing Keras index only.")
                pass

            keras_layer_idx += 1
            if advanced_manual:
                manual_layer_idx +=1
        
        if manual_layer_idx < len(self.layers):
            unassigned_weight_layers_count = 0
            for i in range(manual_layer_idx, len(self.layers)):
                layer_to_check = self.layers[i]
                if isinstance(layer_to_check, (Conv2DLayer, DenseLayer)):
                    unassigned_weight_layers_count +=1
            if unassigned_weight_layers_count > 0:
                print(f"Warning: {unassigned_weight_layers_count} weight-bearing manual layers were not assigned weights from the Keras model. "
                      "This might be due to architecture mismatch or Keras having more layers (InputLayer) "
                      "that were skipped without a corresponding advancement in manual layers.")
                
        print("Weight loading attempt finished.")

