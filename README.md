# Project Description

## Goal
This is the project description for Group 140 in the 02476 Machine Learning Operations course at DTU. The overall goal of the project is to apply the material learned throughout the course to a realistic machine learning problem, with a strong focus on MLOps principles such as reproducibility, experiment tracking, configuration management, and deployment. The project will serve as the basis for the final exam and will be documented through code and a written report.

## Framework
The framework chosen for this project is PyTorch, with the TIMM (PyTorch Image Models) library used for computer vision. TIMM provides access to a wide range of pretrained convolutional neural networks (CNNs), which allows us to efficiently fine-tune existing models instead of training from scratch. The framework will be integrated into a structured codebase with version control and configuration management.

## Data
The dataset used in this project is the Cars Image Dataset from Kaggle. The dataset consists of labeled images of cars belonging to different car brands, such as Audi, BMW, Toyota, Rolls-Royce, Mercedes-Benz, and others. Each image is associated with a single car brand, making the task a supervised multi-class image classification problem. The dataset provides a realistic and diverse set of car images suitable for training and evaluating convolutional neural networks.  

[Kaggle Cars Image Dataset](https://www.kaggle.com/datasets/kshitij192/cars-image-dataset/data)

## Models
The project focuses on performing an image classification task on the car brand dataset using convolutional neural networks. We expect to use pretrained models from the TIMM framework, such as ResNet, EfficientNet, MobileNet, and ConvNeXt, as baseline and comparison models. The models will be fine-tuned on the dataset, and their performance will be evaluated using standard classification metrics.
