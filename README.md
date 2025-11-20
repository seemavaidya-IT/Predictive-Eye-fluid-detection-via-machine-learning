# Predictive-Eye-fluid-detection-via-machine-learning

👁️ Predictive Eye Fluid Detection via Machine Learning

This repository contains a Machine Learning project focused on predicting eye fluid accumulation using medical imaging and data-driven analysis. Eye fluid build-up is one of the early indicators of retinal diseases such as Diabetic Retinopathy, Macular Edema, and Retinal Detachment. Early detection can help doctors provide timely treatment and prevent vision loss. This project develops an ML model that automates the detection process using modern image processing and machine learning techniques.

📘 Project Overview

This project aims to classify eye images (OCT scans / retinal images) into two categories:

Fluid Detected

No Fluid Detected

The system analyzes eye scan images, extracts features, and predicts the presence of fluid using machine learning classification models. It helps in medical diagnosis support, reducing manual workload, and improving accuracy in screening retinal disorders.

🎯 Objectives

To build an automated system for detecting eye fluid using ML

To process and analyze medical imaging data

To train classification models capable of identifying abnormalities

To evaluate model performance using medical-grade metrics

To support ophthalmologists with early detection tools

🛠 Technologies & Tools Used
Programming Language

Python

Libraries

NumPy, Pandas – data handling

Matplotlib, Seaborn – visualizations

OpenCV / PIL – image preprocessing

Scikit-Learn – ML models & evaluation

TensorFlow / Keras (if deep learning is used)

LabelImg / ImageDataset tools – dataset preparation

Platforms

Jupyter Notebook

Google Colab (optional)

VS Code

📂 Dataset Description

The dataset consists of retinal/OCT eye scan images categorized into:

Fluid Present – presence of intra-retinal or sub-retinal fluid

Fluid Absent – healthy retinal structure with no fluid pockets

Each image is either pre-labeled or manually annotated.

Dataset may include:

OCT B-scan images

RGB retinal fundus images

Noise-reduced diagnostic images

If applicable, mention dataset source:
Kaggle / Retina Dataset / Custom Medical Dataset.

🔧 Workflow & Methodology
1. Data Loading & Understanding

Import image dataset

Analyze labels

Check image dimensions, quality, noise

2. Image Preprocessing

Converting to grayscale (if required)

Resizing images for uniform model input

Removing noise using Gaussian/median filtering

Contrast enhancement using CLAHE

Image normalization

Data augmentation (rotation, flip, zoom) to improve model performance

3. Feature Extraction

Depending on approach:

If Using ML Models:

Extract edges

Texture analysis

HOG (Histogram of Oriented Gradients)

GLCM (Gray-Level Co-Occurrence Matrix)

If Using Deep Learning (CNN):

Automatic feature extraction

Convolution layers

Pooling layers

Flattening

Dense layers for classification

4. Model Training

Trained models may include:

Support Vector Machine (SVM)

Random Forest Classifier

Logistic Regression

K-Nearest Neighbors

CNN Model (Convolutional Neural Network) – most accurate

5. Model Evaluation

Performance metrics include:

Accuracy

Precision

Recall

F1-Score

ROC–AUC Curve

Confusion Matrix

These metrics ensure the model performs well for medical classification.

6. Prediction System

Input: eye scan image

Output: “Fluid Detected” OR “No Fluid Detected”

High probability score

🧪 Results & Key Findings

The model successfully classifies eye images with high accuracy.

CNN provided better performance than traditional ML models.

Noise-reduced and contrast-enhanced images gave higher prediction accuracy.

Early fluid detection aids ophthalmologists during screenings.

🚀 Future Enhancements

Deploy the model using Flask / FastAPI / Streamlit

Build a real-time medical scanner application

Add multi-class detection:

Macular Edema

Diabetic Retinopathy

Retinal Detachment

Integrate with mobile or desktop-based diagnostic tools

Apply transfer learning using ResNet, VGG16, EfficientNet

📘 Conclusion

This project demonstrates the power of machine learning in medical imaging. By detecting eye fluid early, the system assists healthcare professionals in diagnosing retinal diseases more accurately and efficiently. The project showcases a complete workflow of data preprocessing, feature extraction, model training, and evaluation—making it an excellent ML-based healthcare application.
