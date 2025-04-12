
# 🗑️ Waste Segregation using CNN

A Convolutional Neural Network (CNN)-based solution for automatic waste classification into categories like biodegradable, non-biodegradable, recyclable, and hazardous. The goal of this project is to contribute to efficient waste management by enabling automated, real-time segregation of waste using computer vision.

## 📌 Problem Statement

Manual waste segregation is time-consuming, labor-intensive, and prone to human error. This project aims to automate the waste classification process using deep learning, specifically CNNs, which are well-suited for image-based tasks.

---

## 🚀 Features

- Classifies waste images into predefined categories  
- Uses CNN architecture built with TensorFlow/Keras  
- Achieves high accuracy with minimal preprocessing  
- Visualizes training progress (loss/accuracy curves)  
- Easy-to-use and modular code  

---

## 📂 Dataset

You can use a publicly available dataset such as:

- C:\Users\Sanskruti\Downloads\Waste Segregation 
  Contains 2,527 images of waste, labeled as:
  - cardboard  
  - glass  
  - metal  
  - paper  
  - plastic  
  - trash  

> You can also expand this by collecting your own dataset with categories such as biodegradable, non-biodegradable, recyclable, etc.

---

## 🧠 Model Architecture

A simple CNN architecture with the following layers:
- Convolutional layers with ReLU activation
- MaxPooling layers for dimensionality reduction
- Fully connected dense layers
- Dropout for regularization
- Softmax output layer for classification

You can customize the architecture in `model.py`.

---

## 🛠️ Installation and Setup

1. Clone the repo:
```bash
git clone https://github.com/yourusername/waste-segregation-cnn.git
cd waste-segregation-cnn
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) Download and prepare the dataset:
```bash
python preprocess.py
```

---

## 📊 Training

To train the model on your dataset:
```bash
python train.py
```

To monitor training using TensorBoard:
```bash
tensorboard --logdir=logs/
```

---

## 📈 Evaluation

Evaluate the model’s performance:
```bash
python evaluate.py
```

Metrics such as accuracy, precision, recall, and confusion matrix will be shown.

---

## 🔍 Inference

To classify a new image:
```bash
python predict.py --image path_to_image.jpg
```

---

## 📌 Future Work

- Improve model with transfer learning (e.g., MobileNet, ResNet)  
- Deploy model on edge devices (like Raspberry Pi)  
- Build a web interface for user interaction  
- Real-time camera-based waste detection  

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you’d like to change.

---

## 🙋‍♀️ Acknowledgements
- TensorFlow & Keras for model building  
- Matplotlib and scikit-learn for visualization and evaluation  

