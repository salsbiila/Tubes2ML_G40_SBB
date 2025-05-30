import numpy as np
import tensorflow as tf
from tensorflow import keras
from from_scratch.layer_embedding import EmbeddingLayer
from from_scratch.layer_simpleRNN import SimpleRNNLayer
from from_scratch.layer_bidirectional import BidirectionalRNNLayer
from from_scratch.layer_dense import DenseLayer


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
        
        layer_idx = 0
        for i, layer in enumerate(self.model.layers):
            layer_type = layer.__class__.__name__
            print(f"Layer {i}: {layer_type} - {layer.name}")
            
            if layer_type == 'Embedding':
                weights = layer.get_weights()
                if len(weights) > 0:
                    embedding_weights = weights[0]
                    print(f"  Embedding weights shape: {embedding_weights.shape}")
                    self.layers.append(EmbeddingLayer(embedding_weights))
                    layer_idx += 1
            
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
                    raise ValueError(f"Unexpected number of weights in bidirectional layer: {len(all_weights)}")
                
                print(f"  Forward weights shapes: {[w.shape for w in forward_weights]}")
                print(f"  Backward weights shapes: {[w.shape for w in backward_weights]}")

                merge_mode = 'concat'
                if hasattr(layer, 'merge_mode'):
                    merge_mode = layer.merge_mode
                
                self.layers.append(BidirectionalRNNLayer(forward_weights, backward_weights, merge_mode))
                layer_idx += 1
            
            elif layer_type == 'SimpleRNN':
                weights = layer.get_weights()
                print(f"  SimpleRNN weights shapes: {[w.shape for w in weights]}")
                
                return_seq = getattr(layer, 'return_sequences', False)
                go_back = getattr(layer, 'go_backwards', False)
                
                self.layers.append(SimpleRNNLayer(weights, return_sequences=return_seq, go_backwards=go_back))
                layer_idx += 1
            
            elif layer_type == 'Dense':
                weights = layer.get_weights()
                if len(weights) == 0:
                    print(f"  Warning: Dense layer has no weights!")
                    continue
                
                dense_weights = weights[0]
                dense_bias = weights[1] if len(weights) > 1 else None
                
                print(f"  Dense weights shape: {dense_weights.shape}")
                if dense_bias is not None:
                    print(f"  Dense bias shape: {dense_bias.shape}")
                
                # Get activation function
                activation = self._get_activation_name(layer.activation)
                print(f"  Dense activation: {activation}")
                
                self.layers.append(DenseLayer(dense_weights, dense_bias, activation=activation))
                layer_idx += 1
            
            elif layer_type == 'Dropout':
                print("  Skipping Dropout layer (inference mode)")
                continue
            
            else:
                print(f"  Warning: Unknown layer type {layer_type}, skipping...")
                continue
        
        print(f"Built {len(self.layers)} functional layers")
    
    def forward(self, x):
        print(f"Input shape: {x.shape}")
        current_output = x
        
        for i, layer in enumerate(self.layers):
            print(f"Processing layer {i}: {type(layer).__name__}")
            current_output = layer.forward(current_output)
            print(f"  Output shape: {current_output.shape}")
        
        return current_output
    
    def predict(self, x):
        probabilities = self.forward(x)
        return np.argmax(probabilities, axis=1)

def debug_model_structure(keras_model_path):
    print("=== MODEL STRUCTURE DEBUG ===")
    
    model = keras.models.load_model(keras_model_path)
    print(f"Model type: {type(model)}")
    print(f"Number of layers: {len(model.layers)}")
    print()
    
    for i, layer in enumerate(model.layers):
        print(f"Layer {i}: {layer.__class__.__name__}")
        print(f"  Name: {layer.name}")
        print(f"  Input shape: {layer.input_shape if hasattr(layer, 'input_shape') else 'N/A'}")
        print(f"  Output shape: {layer.output_shape if hasattr(layer, 'output_shape') else 'N/A'}")

        weights = layer.get_weights()
        print(f"  Weights count: {len(weights)}")
        for j, w in enumerate(weights):
            print(f"    Weight {j} shape: {w.shape}")

        if hasattr(layer, 'activation'):
            print(f"  Activation: {layer.activation}")
        if hasattr(layer, 'return_sequences'):
            print(f"  Return sequences: {layer.return_sequences}")
        if hasattr(layer, 'go_backwards'):
            print(f"  Go backwards: {layer.go_backwards}")
        if hasattr(layer, 'merge_mode'):
            print(f"  Merge mode: {layer.merge_mode}")
        
        print()

def compare_implementations(keras_model_path, test_data, test_labels, max_samples=400):
    if hasattr(test_data, 'numpy'):
        test_data = test_data.numpy()
    if hasattr(test_labels, 'numpy'):
        test_labels = test_labels.numpy()

    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

    if len(test_data) > max_samples:
        indices = np.random.choice(len(test_data), max_samples, replace=False)
        test_data_subset = test_data[indices]
        test_labels_subset = test_labels[indices]
    else:
        test_data_subset = test_data
        test_labels_subset = test_labels
    
    print(f"Using {len(test_data_subset)} samples for comparison")
    

    keras_model = keras.models.load_model(keras_model_path)
    scratch_model = SimpleRNNModelFromScratch(keras_model_path)
    print("\nGetting Keras predictions...")
    keras_predictions = keras_model.predict(test_data_subset, verbose=0)
    keras_classes = np.argmax(keras_predictions, axis=1)
    print("\nGetting from-scratch predictions...")
    batch_size = 8
    scratch_predictions_list = []
    
    for i in range(0, len(test_data_subset), batch_size):
        end_idx = min(i + batch_size, len(test_data_subset))
        batch = test_data_subset[i:end_idx]
        print(f"  Batch {i//batch_size + 1}/{(len(test_data_subset)-1)//batch_size + 1}")

        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            batch_pred = scratch_model.forward(batch)
            scratch_predictions_list.append(batch_pred)
        finally:
            sys.stdout = old_stdout
    
    scratch_predictions = np.vstack(scratch_predictions_list)
    scratch_classes = np.argmax(scratch_predictions, axis=1)

    print(f"\nKeras predictions shape: {keras_predictions.shape}")
    print(f"Scratch predictions shape: {scratch_predictions.shape}")
    
    from sklearn.metrics import f1_score, accuracy_score
    
    keras_f1 = f1_score(test_labels_subset, keras_classes, average='macro')
    scratch_f1 = f1_score(test_labels_subset, scratch_classes, average='macro')
    
    keras_acc = accuracy_score(test_labels_subset, keras_classes)
    scratch_acc = accuracy_score(test_labels_subset, scratch_classes)
    
    prediction_agreement = np.mean(keras_classes == scratch_classes)

    max_diff = np.max(np.abs(keras_predictions - scratch_predictions))
    mean_diff = np.mean(np.abs(keras_predictions - scratch_predictions))
    
    print(f"\n====================== RESULTS =======================")
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