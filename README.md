# Emotion_Detection_from_Faces

🧠 Project Overview
This project detects human emotions—Happy, Sad, and Angry—from facial images using a Convolutional Neural Network (CNN). It uses OpenCV for real-time face detection and Keras for model training.
📂 Dataset
Name: FER-2013
Source: https://www.kaggle.com/datasets/msambare/fer2013
Format: 48x48 grayscale facial images with emotion labels
Used Classes: Happy, Sad, Angry (filtered from 7 total classes)
🛠️ Tools & Technologies
- OpenCV – Face detection and image capture
- TensorFlow / Keras – CNN model training
- Pandas / NumPy / Matplotlib – Data processing and visualization
🔧 Installation Steps
1. Clone the Repository:
   git clone https://github.com/yourusername/emotion-detection-faces.git
2. Navigate to the Folder:
   cd emotion-detection-faces
3. Install Required Libraries:
   pip install -r requirements.txt
4. Download the FER-2013 Dataset and place fer2013.csv in the data/ folder.
🏗️ Project Structure

emotion-detection-faces/
│
├── data/
│   └── fer2013.csv
├── models/
│   └── emotion_model.h5
├── notebooks/
│   └── EDA_and_Model_Training.ipynb
├── app/
│   └── detect_emotion.py
├── utils/
│   └── preprocess.py
├── README.docx
└── requirements.txt

🧠 Model Architecture
Input: 48x48 grayscale image
Layers:
- Conv2D → ReLU → MaxPooling
- Conv2D → ReLU → MaxPooling
- Flatten → Dense → Dropout
- Output Layer: Softmax (3 emotion classes)
🚀 How to Use
1. Train the Model:
   python train.py
2. Start Webcam Detection:
   python app/detect_emotion.py
📊 Performance
Metric: Accuracy ≈ 85% (filtered classes)
Validation Loss: Low after 20 epochs
🖼️ Sample Output
Real-time face detection with emotion label overlaid:
- Bounding box around face
- Detected emotion displayed above face
✅ Future Improvements
- Add support for all 7 emotions
- Improve accuracy with data augmentation
- Deploy as a Flask or Streamlit web app
Model Accuracy:
<img width="576" height="455" alt="Model_accuracy" src="https://github.com/user-attachments/assets/5be8e2d7-ac09-4583-b809-d4685065eadb" />

Model Loss:
<img width="567" height="455" alt="Model_loss" src="https://github.com/user-attachments/assets/aebf3c9f-a888-4ae0-814f-178d9c8ec0da" />

