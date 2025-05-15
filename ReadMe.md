# Garbage Classification System using CNN & LSTM

This project is a real-time garbage classification system that uses deep learning (CNN and LSTM) to detect and classify types of waste from a live video stream. The system identifies six types of garbage: **Plastic, Cardboard, Glass, Metal, Paper, and Trash**.

## Table of Contents

- [Features](#-features)
- [Technologies Used](#️technologies-used)
- [Garbage Classes](#-garbage-classes)
- [How to Run the Project](#️how-to-run-the-project)
- [Future Improvements](#-future-improvements)
## Features

- Real-time classification using webcam feed
- Trained with a CNN + optional LSTM hybrid model
- Intuitive and modern web interface with live video stream
- Data preprocessing and model training scripts included
- Flask-based backend
- Responsive UI with gradient background and control buttons

---

## Technologies Used

| Category        | Technology             |
|----------------|------------------------|
| Frontend       | HTML, CSS, JavaScript  |
| Backend        | Python, Flask          |
| Deep Learning  | TensorFlow, Keras      |
| Computer Vision| OpenCV                 |
| Model Types    | CNN, LSTM              |

---

## Garbage Classes

- Plastic  
- Cardboard  
- Glass  
- Metal  
- Paper  
- Trash  

---

## How to Run the Project

```
1. Clone the Repository
git clone https://github.com/your-username/garbage-classification.git
cd garbage-classification

2. Install Dependencies
pip install -r requirements.txt
Make sure your webcam is enabled and accessible.

3. Download the Dataset from Kaggle
This project uses the Garbage Classification Dataset from Kaggle.
Download it and unzip the file in the root location.

4. Train the Model (CNN)
Use the provided script to train the CNN and LSTM model:
python train_cnn.py
The trained model will be saved as cnn_model.h5
python train_lstm.py
The trained model will be saved as lstm_model.h5

5. Run the Web App
python app.py
Then open your browser and go to:
📍 http://127.0.0.1:5000
```


## Future Improvements

Integrate YOLO for object detection before classification
Deploy on cloud (e.g., Heroku, AWS)
Add database for storing prediction history
Mobile app version using TensorFlow Lite

