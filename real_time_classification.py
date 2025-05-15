import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained CNN model
cnn_model = load_model('cnn_model.h5')

# Define the labels based on training data
labels = ["Plastic", "Cardboard", "Glass", "Metal", "Paper", "Trash"]

# Function to preprocess the frame for the CNN model
def preprocess_for_cnn(frame):
    img = cv2.resize(frame, (150, 150))  # Adjust to match your CNN model's input
    img = img / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Shape: (1, 150, 150, 3)
    return img

def start_classification():
    cap = cv2.VideoCapture(0)  # Capture from the webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess the frame for CNN input
        cnn_input = preprocess_for_cnn(frame)
        
        # Predict using the CNN
        cnn_pred = cnn_model.predict(cnn_input)
        predicted_label = labels[np.argmax(cnn_pred)]

        # Display the predicted label on the frame
        cv2.putText(frame, f"Detected: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw a bounding box around the area of interest
        h, w, _ = frame.shape
        top_left = (int(w * 0.3), int(h * 0.3))
        bottom_right = (int(w * 0.7), int(h * 0.7))
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

def start_classification():
    return generate_frames()
