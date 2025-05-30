import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt

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


def build_simple_rnn_model(vectorize_layer, embedding_dim=128, rnn_units=64, 
                          dropout_rate=0.5, num_classes=3):
    vocab_size = vectorize_layer.vocabulary_size()
    
    model = keras.Sequential([
        layers.Embedding(vocab_size, embedding_dim),
        layers.Bidirectional(layers.SimpleRNN(rnn_units, return_sequences=False)),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model

def model_multi_layers(vectorize_layer, embedding_dim=128):
    vocab_size = vectorize_layer.vocabulary_size()
    
    model = keras.Sequential([
        layers.Embedding(vocab_size, embedding_dim),
        layers.Bidirectional(layers.SimpleRNN(64, return_sequences=True)),
        layers.Bidirectional(layers.SimpleRNN(32, return_sequences=False)),
        layers.Dropout(0.5),
        layers.Dense(3, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', 
                 optimizer='adam', metrics=['accuracy'])
    return model


def train_model(model, train_data, train_labels, valid_data, valid_labels, 
                epochs=20, batch_size=32):
    history = model.fit(
        train_data, train_labels,
        validation_data=(valid_data, valid_labels),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    return history

def evaluate_model(model, test_data, test_labels, label_mapping):
    predictions = model.predict(test_data)
    predicted_classes = np.argmax(predictions, axis=1)

    macro_f1 = f1_score(test_labels, predicted_classes, average='macro')

    label_names = list(label_mapping.keys())

    print("\nClassification Report:")
    print(classification_report(test_labels, predicted_classes, target_names=label_names))
    
    return macro_f1, predicted_classes

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    print("Loading data...")
    train_df, valid_df, test_df = load_data()
    
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(valid_df)}")
    print(f"Test samples: {len(test_df)}")

    print("\nPreprocessing data...")
    max_tokens = 10000
    sequence_length = 100
    
    (train_sequences, train_labels, valid_sequences, valid_labels, 
     test_sequences, test_labels, vectorize_layer, label_mapping) = preprocess_data(train_df, valid_df, test_df, max_tokens, sequence_length)
    
    vocab_size = vectorize_layer.vocabulary_size()
    print(f"\nVocabulary size: {vocab_size}")
    print(f"Sequence length: {sequence_length}")

    print("\nBuilding Simple RNN model...")

    model = build_simple_rnn_model(
        vectorize_layer=vectorize_layer,
        embedding_dim=128,
        rnn_units=64,
        dropout_rate=0.5,
        num_classes=3
    )
    # model = model_multi_layers(vectorize_layer)
    
    model.summary()

    print("\nTraining model...")
    history = train_model(
        model, 
        train_sequences, train_labels,
        valid_sequences, valid_labels,
        epochs=20,
        batch_size=32
    )

    plot_training_history(history)

    print("\nEvaluating model on test set...")
    macro_f1, predictions = evaluate_model(model, test_sequences, test_labels, label_mapping)
    print(f"\nMacro F1-Score: {macro_f1:.4f}")

    print("\nSaving model and weights...")
    model.save('model/simple_rnn_model.h5')
    model.save_weights('model/simple_rnn_weights.h5')

    vectorize_layer_config = vectorize_layer.get_config()
    vectorize_layer_weights = vectorize_layer.get_weights()
    
    import pickle
    with open('model/vectorization_config.pickle', 'wb') as f:
        pickle.dump(vectorize_layer_config, f)
    with open('model/vectorization_weights.pickle', 'wb') as f:
        pickle.dump(vectorize_layer_weights, f)
    
    print("\nTraining completed!")
    
    return model, history, vectorize_layer

if __name__ == "__main__":
    model, history, vectorize_layer = main()