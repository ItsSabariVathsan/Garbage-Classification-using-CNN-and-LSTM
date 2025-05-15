import numpy as np
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Directory where your dataset is stored
data_dir = 'E:\\College Files\\Projects\\Garbage Classification\\FINAL PROJECT\\Garbage classification'

# Example dataset properties with reduced time_steps and image_size
time_steps = 5  # Reduced time steps
image_size = (128, 128)  # Reduced image size to save memory
num_classes = 6

# Prepare sequences and labels
X_train_sequences = []
y_train_sequences = []
class_folders = ['CARDBOARD', 'GLASS', 'METAL', 'PLASTIC', 'PAPER', 'TRASH']

for class_idx, class_folder in enumerate(class_folders):
    class_path = os.path.join(data_dir, class_folder)
    image_files = [f for f in os.listdir(class_path) if f.endswith('.jpg')]
    
    for i in range(0, len(image_files) - time_steps, time_steps):
        sequence = []
        for j in range(i, i + time_steps):
            img_path = os.path.join(class_path, image_files[j])
            img = cv2.imread(img_path)
            img = cv2.resize(img, image_size)
            img = img / 255.0  # Normalize
            sequence.append(img)
        X_train_sequences.append(np.array(sequence))
        y_train_sequences.append(class_idx)

# Convert to numpy arrays
X_train_sequences = np.array(X_train_sequences)  # Shape: (num_sequences, time_steps, 128, 128, 3)
y_train_sequences = np.array(y_train_sequences)  # Shape: (num_sequences,)

# One-hot encode labels
y_train_sequences = to_categorical(y_train_sequences, num_classes=num_classes)

# Define LSTM model with smaller architecture
lstm_model = Sequential([
    TimeDistributed(Conv2D(16, (3, 3), activation='relu'), input_shape=(None, 128, 128, 3)),  # Smaller CNN layers
    TimeDistributed(MaxPooling2D((2, 2))),
    TimeDistributed(Flatten()),
    LSTM(20, return_sequences=False),  # Smaller LSTM layer
    Dense(num_classes, activation='softmax')
])

lstm_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with reduced batch size
lstm_history = lstm_model.fit(
    X_train_sequences, 
    y_train_sequences, 
    epochs=10, 
    validation_split=0.2,
    batch_size=8  # Smaller batch size to fit in memory
)

# Save the model
lstm_model.save('lstm_model.h5')
