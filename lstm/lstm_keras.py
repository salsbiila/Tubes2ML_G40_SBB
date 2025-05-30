import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt
import pickle
import os

np.random.seed(42)
tf.random.set_seed(42)

def load_data():
    train_df = pd.read_csv('data/train.csv')
    valid_df = pd.read_csv('data/valid.csv')
    test_df = pd.read_csv('data/test.csv')
    
    return train_df, valid_df, test_df

def preprocess_data(train_df, valid_df, test_df, max_tokens=10000, sequence_length=100):
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

def build_lstm_model_single_layer(vectorize_layer, embedding_dim=128, lstm_units=64, dropout_rate=0.5, num_classes=3, bidirectional=True):
    vocab_size = vectorize_layer.vocabulary_size()
    
    model = keras.Sequential([
        layers.Embedding(vocab_size, embedding_dim, name='embedding_layer'),
        layers.Bidirectional(
            layers.LSTM(lstm_units, return_sequences=False, name='lstm_layer')
        ) if bidirectional else layers.LSTM(lstm_units, return_sequences=False, name='lstm_layer'),
        layers.Dropout(dropout_rate, name='dropout_layer'),
        layers.Dense(num_classes, activation='softmax', name='output_layer')
    ])

    model.compile(
        loss='sparse_categorical_crossentropy',  
        optimizer='adam', 
        metrics=['accuracy']
    )
    
    return model

def build_multi_layer_lstm(vectorize_layer, embedding_dim=128, lstm_units_list=[64, 32], dropout_rate=0.2, num_classes=3, bidirectional=True):
    vocab_size = vectorize_layer.vocabulary_size()
    
    model = keras.Sequential()
    
    model.add(layers.Embedding(vocab_size, embedding_dim, name='embedding_layer'))
    
    for i, units in enumerate(lstm_units_list):
        return_sequences = True
        if (i == len(lstm_units_list) - 1) :
            return_sequences = False

        if bidirectional:
            model.add(layers.Bidirectional(
                layers.LSTM(units, return_sequences=return_sequences, name=f'lstm_layer_{i+1}')
            ))
        else:
            model.add(layers.LSTM(units, return_sequences=return_sequences, name=f'lstm_layer_{i+1}'))
        
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate, name=f'dropout_layer_{i+1}'))
    
    model.add(layers.Dense(num_classes, activation='softmax', name='output_layer'))
    
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, train_data, train_labels, valid_data, valid_labels, epochs=20, batch_size=32, verbose=1):
    history = model.fit(
        train_data, train_labels,
        validation_data=(valid_data, valid_labels),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose
    )
    
    return history

def evaluate_model(model, test_data, test_labels, label_mapping):
    predictions = model.predict(test_data, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)

    macro_f1 = f1_score(test_labels, predicted_classes, average='macro')

    label_names = list(label_mapping.keys())

    print("\nClassification Report:")
    print(classification_report(test_labels, predicted_classes, target_names=label_names))
    
    return macro_f1, predicted_classes

def plot_training_history(history, title="LSTM Model Training History"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title(f'{title} - Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    print("Loading data...")
    train_df, valid_df, test_df = load_data()
    
    print(f"Data loaded successfully!")
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(valid_df)}")
    print(f"Test samples: {len(test_df)}")

    print("\nPreprocessing data...")
    max_tokens = 10000
    sequence_length = 100
    
    (train_sequences, train_labels, valid_sequences, valid_labels, test_sequences, test_labels, vectorize_layer, label_mapping) = preprocess_data(train_df, valid_df, test_df, max_tokens, sequence_length)
    
    vocab_size = vectorize_layer.vocabulary_size()
    print(f"Preprocessing completed!")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Sequence length: {sequence_length}")
    print(f"Label mapping: {label_mapping}")

    model = build_lstm_model_single_layer(
        vectorize_layer=vectorize_layer,
        embedding_dim=50,
        lstm_units=32,
        dropout_rate=0.5, 
        num_classes=len(label_mapping) 
    )
    
    print("Model built successfully!")
    model.summary()

    print("\nTraining LSTM model...")
    history = train_model(
        model, 
        train_sequences, train_labels,
        valid_sequences, valid_labels,
        epochs=20,
        batch_size=32
    )
    print("Training completed!")

    print("\nPlotting training history...")
    plot_training_history(history)

    print("\nEvaluating model on test set...")
    macro_f1, predictions = evaluate_model(model, test_sequences, test_labels, label_mapping)

    print(f"\nFinal Macro F1-Score: {macro_f1:.4f}")

    print("\nSaving model and weights...")
    model.save('model/lstm_model.h5')
    model.save_weights('model/lstm.weights.h5')

    vectorize_layer_config = vectorize_layer.get_config()
    vectorize_layer_weights = vectorize_layer.get_weights()
    
    import pickle
    with open('model/vectorization_config.pickle', 'wb') as f:
        pickle.dump(vectorize_layer_config, f)
    with open('model/vectorization_weights.pickle', 'wb') as f:
        pickle.dump(vectorize_layer_weights, f)

    print("\nModel Saved")
    plot_training_history(history)

    return model, history, vectorize_layer
        
if __name__ == "__main__":
    model, history, vectorize_layer = main()