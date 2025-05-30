import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
import pickle
from from_scratch.LSTM_scratch import debug_lstm_model_structure, compare_lstm_implementations

def preprocess_data_exact_copy(train_df, valid_df, test_df, max_tokens=10000, sequence_length=100):
    train_texts = train_df['text'].values
    train_labels = train_df['label'].values
    valid_texts = valid_df['text'].values
    valid_labels = valid_df['label'].values
    test_texts = test_df['text'].values
    test_labels = test_df['label'].values

    label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}

    train_labels = np.array([label_mapping[label] for label in train_labels])
    valid_labels = np.array([label_mapping[label] for label in valid_labels])
    test_labels = np.array([label_mapping[label] for label in test_labels])

    vectorize_layer = layers.TextVectorization(
        max_tokens=max_tokens,
        output_sequence_length=sequence_length,
        output_mode='int'  
    )

    vectorize_layer.adapt(train_texts)

    train_sequences = vectorize_layer(train_texts)
    valid_sequences = vectorize_layer(valid_texts)
    test_sequences = vectorize_layer(test_texts)
    
    return (train_sequences, train_labels, valid_sequences, valid_labels, 
            test_sequences, test_labels, vectorize_layer, label_mapping)

def full_test():
    print("1. Print LSTM model structure")
    debug_lstm_model_structure('model/lstm_model.h5')

    print("2. Loading all data (same as training)")
    train_df = pd.read_csv('data/train.csv')
    valid_df = pd.read_csv('data/valid.csv')
    test_df = pd.read_csv('data/test.csv')

    print("3. Using EXACT same preprocessing as training script")
    (train_sequences, train_labels, valid_sequences, valid_labels, 
     test_sequences, test_labels, vectorize_layer, label_mapping) = preprocess_data_exact_copy(
        train_df, valid_df, test_df, 
        max_tokens=10000,    
        sequence_length=100    
    )
    
    print(f"Vocabulary size: {vectorize_layer.vocabulary_size()}")
    print(f"Test data shape: {test_sequences.shape}")
    print(f"Test labels shape: {test_labels.shape}")

    print("\n4. Comparing LSTM implementations")
    results = compare_lstm_implementations(
        keras_model_path='model/lstm_model.h5',
        test_data=test_sequences,
        test_labels=test_labels,
        max_samples=400
    )    
    return results

if __name__ == "__main__":
    print("=== LSTM TESTING (EXACT TRAINING PREPROCESSING) ===")
    results = full_test()
    
    if results:
        print(f"SUCCESS!")
        print(f"Prediction Agreement: {results['prediction_agreement']:.4f}")
        print(f"Keras F1-Score: {results['keras_f1']:.4f}")
        print(f"Scratch F1-Score: {results['scratch_f1']:.4f}")
        print(f"Max Difference: {results['max_diff']:.6f}")
    else:
        print("❌ Test failed!")