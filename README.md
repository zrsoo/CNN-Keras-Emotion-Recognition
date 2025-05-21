# CNN-Keras Emotion Recognition 🧠 👁️ 😊

A deep learning project that recognizes human emotions from facial expressions using Convolutional Neural Networks (CNNs) with Keras. This repository contains both a custom CNN implementation and transfer learning experiments with MobileNetV2.

## 📋 Overview

This project explores the fascinating challenge of emotion recognition from facial images using deep learning techniques. The system can identify seven basic human emotions (angry, disgusted, fearful, happy, neutral, sad, surprised) from facial expressions captured via webcam in real-time.

The repository includes:

- 🔍 A custom CNN model built from scratch using Keras
- 🔄 Transfer learning experiments with MobileNetV2
- 📹 Real-time emotion recognition from webcam feed
- 📊 Performance analysis and comparison of approaches

## 🛠️ Technologies Used

- **Python** - Primary programming language
- **Keras/TensorFlow** - Deep learning framework
- **OpenCV** - Computer vision library for face detection and image processing
- **NumPy** - Numerical operations
- **Matplotlib** - Visualization of training metrics
- **Jupyter Notebook** - Interactive development for transfer learning experiments

## 🧩 Project Components

### 1. Custom CNN Model 🧠

The custom CNN model is defined in `train.py` using Keras Sequential API with the following architecture:

```
Input (48x48 grayscale images)
│
├─ Conv2D (32 filters, 3x3 kernel, ReLU)
│
├─ Conv2D (64 filters, 3x3 kernel, ReLU)
│
├─ MaxPooling2D (2x2)
│
├─ Dropout (0.25)
│
├─ Conv2D (128 filters, 3x3 kernel, ReLU)
│
├─ MaxPooling2D (2x2)
│
├─ Conv2D (128 filters, 3x3 kernel, ReLU)
│
├─ MaxPooling2D (2x2)
│
├─ Dropout (0.25)
│
├─ Flatten
│
├─ Dense (1024 units, ReLU)
│
├─ Dropout (0.5)
│
└─ Dense (7 units, Softmax) → Output (7 emotion classes)
```

### 2. Transfer Learning with MobileNetV2 🔄

The `jupyter/transfer_learning.ipynb` notebook explores using MobileNetV2 (pre-trained on ImageNet) for emotion recognition:

- Base MobileNetV2 model with frozen weights
- Custom top layers for emotion classification
- Data augmentation techniques
- Feature extraction and fine-tuning experiments

### 3. Real-time Emotion Recognition 📹

The `test.py` script implements a real-time emotion recognition system:

1. Captures video from webcam
2. Detects faces using Haar Cascade classifier
3. For each detected face:
   - Extracts and preprocesses the face region
   - Feeds it to the trained CNN model
   - Displays the predicted emotion on the video feed

## 📊 Training Details

### Dataset

- **FER-2013** (Facial Expression Recognition 2013)
- 48x48 pixel grayscale images of faces
- Seven emotion categories: angry, disgusted, fearful, happy, neutral, sad, surprised
- Data organized in `data/train` and `data/test` directories with subdirectories for each emotion class

### Preprocessing & Augmentation

- Images rescaled (divided by 255)
- Real-time augmentation during training:
  - Random rotations (±0.1 degrees)
  - Random horizontal flips

### Training Parameters

- **Optimizer**: Adam (learning rate = 0.0001, decay = 1e-6)
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 50
- **Batch Size**: 64

## 📈 Results

- The custom CNN model achieved approximately **63.4%** accuracy on the test set
- The transfer learning approach with MobileNetV2 achieved around **35%** accuracy
- The custom CNN outperformed the transfer learning approach for this specific task

## 🚀 Usage

### Prerequisites

- Python 3.6+
- Required libraries: TensorFlow, Keras, OpenCV, NumPy, Matplotlib

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/zrsoo/CNN-Keras-Emotion-Recognition.git
   cd CNN-Keras-Emotion-Recognition
   ```

2. Install required dependencies:
   ```
   pip install tensorflow opencv-python numpy matplotlib
   ```

### Training the Model

To train the custom CNN model:

```
cd PythonER
python train.py
```

This will:
- Load the FER-2013 dataset
- Train the CNN model
- Save the model architecture to `emodel.json`
- Save the model weights to `emodel.h5`
- Generate accuracy and loss plots

### Running Real-time Emotion Recognition

To run the real-time emotion recognition system:

```
cd PythonER
python test.py
```

This will:
- Load the trained model
- Access your webcam
- Detect faces and predict emotions in real-time
- Press 'q' to quit the application

## 📁 Repository Structure

```
ER-CNN-Keras/
├── Documentation.pdf           # Detailed project report
├── PythonER/                   # Code for the custom CNN model
│   ├── haarcascade/            # Haar cascade file for face detection
│   │   └── haarcascade_frontalface_default.xml
│   ├── script/                 # Additional scripts (if any)
│   │   └── script.py
│   ├── test.py                 # Real-time emotion recognition script
│   ├── train.py                # Training script for the custom CNN model
│   ├── modelaccuracy.png       # Plot of CNN training/validation accuracy
│   ├── modelloss.png           # Plot of CNN training/validation loss
│   └── mobilenetv2accuracy.png # Plot related to MobileNetV2 experiments
├── jupyter/                    # Jupyter notebook for transfer learning
│   └── transfer_learning.ipynb
├── poster.pptx                 # Project poster presentation
└── teaser.mp4                  # Project teaser video
```

## 🔍 Future Improvements

- Explore more advanced architectures like ResNet or EfficientNet
- Implement ensemble methods combining multiple models
- Add more data augmentation techniques to improve generalization
- Optimize for mobile deployment
- Explore emotion recognition in different lighting conditions and angles

## 📚 Resources

- [FER-2013 Dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
- [Keras Documentation](https://keras.io/)
- [OpenCV Documentation](https://docs.opencv.org/)

---

Created by [zrsoo](https://github.com/zrsoo) 💻
