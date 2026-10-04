from pathlib import Path

# Project path
BASE_DIR = Path(__file__).resolve().parent

# Dataset path
DATASET_PATH = BASE_DIR / "dataset" / "shakespeare.txt"

# Load Shakespeare text
text = DATASET_PATH.read_text(encoding="utf-8")

print("Dataset loaded successfully!")
print("Number of characters:", len(text))

print("\nFirst 500 characters:")
print(text[:500])
import string

# Convert text to lowercase
text = text.lower()

# Remove punctuation
text = text.translate(str.maketrans("", "", string.punctuation))

print("\nPreprocessing completed!")
print("First 500 characters after preprocessing:")
print(text[:500])
from tensorflow.keras.preprocessing.text import Tokenizer

# Create tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

# Convert words to integer IDs
total_words = len(tokenizer.word_index) + 1

print("\nTokenization completed!")
print("Total unique words:", total_words)
print("Sample word-to-index mapping:")
print(list(tokenizer.word_index.items())[:20])
# Convert the entire text into integer sequences
input_sequences = []

# Split text into lines
for line in text.split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]

    # Create sequences of increasing length
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

print("\nInput-output sequences created!")
print("Total sequences:", len(input_sequences))
print("Example sequence:", input_sequences[0])
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Find the longest sequence
max_sequence_len = max(len(seq) for seq in input_sequences)

# Pad all sequences to the same length
input_sequences = np.array(
    pad_sequences(input_sequences, maxlen=max_sequence_len, padding="pre")
)

# Split into input (X) and output (y)
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

print("\nPadding completed!")
print("Maximum sequence length:", max_sequence_len)
print("X shape:", X.shape)
print("y shape:", y.shape)
from sklearn.model_selection import train_test_split

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.1,
    random_state=42
)

print("\nTrain-validation split completed!")
print("Training samples:", X_train.shape[0])
print("Validation samples:", X_val.shape[0])
from sklearn.model_selection import train_test_split

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.1,
    random_state=42
)

print("\nTrain-validation split completed!")
print("Training samples:", X_train.shape[0])
print("Validation samples:", X_val.shape[0])
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# LSTM model
model = Sequential([
    Embedding(
        input_dim=total_words,
        output_dim=128
    ),
    LSTM(128),
    Dense(total_words, activation="softmax")
])

# Build the model explicitly
model.build(input_shape=(None, max_sequence_len - 1))

# Display model architecture
model.summary()
# Compile the model
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

print("\nModel compiled successfully!")
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Stop training when validation loss stops improving
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

# Save the best model
checkpoint = ModelCheckpoint(
    "model/best_lstm_model.keras",
    monitor="val_loss",
    save_best_only=True
)

print("\nTraining callbacks configured!")
# Train the LSTM model
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=128,
    callbacks=[early_stopping, checkpoint]
)

print("\nModel training completed!")