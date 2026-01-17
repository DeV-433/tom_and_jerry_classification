# 🐱 Tom & Jerry Image Classifier 🐭

A Deep Learning project that classifies images from the "Tom and Jerry" cartoon into four distinct classes using a Convolutional Neural Network (CNN).

## 📌 Project Overview

This repository contains a Jupyter Notebook (`model.ipynb`) that builds, trains, and evaluates a CNN model to recognize characters and scenes from the show. The project demonstrates an end-to-end Machine Learning pipeline:

1.  **Data Ingestion & Cleaning**: Handling directory structures and filtering specific "challenge" images into a test set.
2.  **Preprocessing**: Image resizing (224x224), normalization, and data augmentation (flips, rotation, zoom, contrast).
3.  **Model Architecture**: A custom Sequential CNN with multiple convolutional blocks, L2 regularization, and Batch Normalization.
4.  **Training**: Optimized using the Adam optimizer, Early Stopping, and Learning Rate Reduction on Plateau.
5.  **Evaluation**: Achieved **~93.75% accuracy** on the designated challenge test set.

## 📂 Dataset

The dataset used for this project is sourced from Kaggle:
👉 **[Tom and Jerry Image Classification Dataset](https://www.kaggle.com/datasets/balabaskar/tom-and-jerry-image-classification)**

### Classes
The model predicts one of the following four labels:
* `jerry` (0)
* `tom` (1)
* `tom_jerry_0` (2)
* `tom_jerry_1` (3)

## 🛠️ Tech Stack

* **Python 3.12+**
* **TensorFlow / Keras** (Deep Learning framework)
* **Pandas** (Data manipulation)
* **Matplotlib** (Visualization)

## 🧠 Model Architecture

The classifier is built using the TensorFlow Keras Sequential API:
* **Input**: 224x224x3 RGB images.
* **Augmentation Layer**: RandomFlip, RandomRotation, RandomZoom, RandomContrast.
* **Convolutional Blocks**: Three blocks featuring `Conv2D` (32, 64, 128 filters), `ReLU` activation, and `MaxPooling2D`.
* **Regularization**: Used `L2` kernel regularization and `BatchNormalization` layers to reduce overfitting.
* **Dense Layers**: A fully connected layer (64 units) followed by the output layer (4 units, Softmax activation).

## 🚀 Performance

* **Training Accuracy**: ~87%
* **Validation Accuracy**: ~86%
* **Test Accuracy (Challenge Set)**: **93.75%**
* **Test Loss**: 0.3054

## 🔧 Installation & Usage

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/tom-jerry-classifier.git](https://github.com/your-username/tom-jerry-classifier.git)
    cd tom-jerry-classifier
    ```
2.  Install dependencies:
    ```bash
    pip install tensorflow pandas matplotlib
    ```
3.  Download the dataset from Kaggle and unzip it. Ensure the root folder is named `tom_and_jerry` in the same directory as the notebook.
4.  Run the notebook:
    ```bash
    jupyter notebook model.ipynb
    ```

## 📈 Results

The notebook generates accuracy and loss plots to visualize the training progress and ensure the model converges without significant overfitting.

--
