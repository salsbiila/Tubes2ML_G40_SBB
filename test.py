import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import pickle
from rnn_scratch import debug_model_structure, compare_implementations

def full_test():

    print("1. Print model structure...")
    debug_model_structure('model/simple_rnn_model.h5')

    print("2. Loading test data...")
    test_df = pd.read_csv('data/test.csv')
    with open('model/vectorization_config.pickle', 'rb') as f:
        vectorize_config = pickle.load(f)
    with open('model/vectorization_weights.pickle', 'rb') as f:
        vectorize_weights = pickle.load(f)
    
    vectorize_layer = layers.TextVectorization.from_config(vectorize_config)
    vectorize_layer.set_weights(vectorize_weights)
    test_texts = test_df['text'].values
    test_labels = test_df['label'].values
    label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
    test_labels_numeric = np.array([label_mapping[label] for label in test_labels])
    test_sequences = vectorize_layer(test_texts)
    print(f"Test data shape: {test_sequences.shape}")
    print(f"Test labels shape: {test_labels_numeric.shape}")

    print("\n3. Comparing implementations...")
    results = compare_implementations(
        keras_model_path='model/simple_rnn_model.h5',
        test_data=test_sequences,
        test_labels=test_labels_numeric,
        max_samples=300
    )    
    return results

if __name__ == "__main__":
    results = full_test()