# Automated-Pneumonia-Detection-from-Chest-X-Ray-Images-using-CNN

## Overview

This project develops a Convolutional Neural Network (CNN) to distinguish between healthy lungs and those affected by pneumonia from chest X-ray images. It's implemented in Python using TensorFlow and Keras. The model is trained, validated, and tested on a dataset containing 5,863 pediatric chest X-ray images, aiming to assist in early and accurate pneumonia detection.

## Dependencies
Ensure you have Python 3.x and the following packages installed:

TensorFlow==2.10.1

Keras Tuner==1.4.7

NumPy==1.26.4

Pandas==2.2.1

Matplotlib==3.8.0

To install the dependencies, run:

pip install -r requirements.txt

## Dataset
The dataset comprises 5,863 labeled chest X-ray images from children aged 1 to 5, obtained from the Guangzhou Women and Children’s Medical Center. The images are categorized into 'Pneumonia' and 'Normal'.

You can download the dataset from the below link.

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## Model Development
The CNN model incorporates several layers, including convolutional, max pooling, dropout, and fully connected layers, to learn and predict pneumonia from X-ray images. The development process involves:

- Data Preprocessing: Rescaling, augmentation, and batch processing of images.
- Model Building: Architecture design with convolutional, pooling, and dropout layers.
- Training: The model is trained with a focus on preventing overfitting using dropout and L2 regularization techniques.
- Hyperparameter Tuning: Bayesian Optimization is used to fine-tune model parameters for optimal performance.
- Evaluation: The trained model's performance is evaluated on a separate test dataset.

## Results
The dropout model achieved the best performance with an accuracy of 88.6% and a loss of 0.31 on the test dataset, outperforming the base and L2 regularization models. Hyperparameter tuning further refined the model, resulting in a final accuracy of 86.2% and a loss of 0.332.

## Conclusion
This project highlights the effectiveness of CNNs in medical image analysis, specifically in automating pneumonia detection from chest X-rays. Future work can explore expanding the dataset, experimenting with more complex architectures, and incorporating additional evaluation metrics.

## Acknowledgements
Special thanks to our team members for their contributions to this project:

- Shehab Aly
- Rosemary Bagiza
- Jennifer Garrett
- Curtis Robinson
