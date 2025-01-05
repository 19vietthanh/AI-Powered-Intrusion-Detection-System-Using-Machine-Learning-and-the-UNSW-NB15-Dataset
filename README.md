# Intrusion Detection System Using SVM on UNSW-NB15 Dataset

## Overview
This project implements an Intrusion Detection System (IDS) using a Support Vector Machine (SVM) classifier on the UNSW-NB15 dataset. The goal is to classify network traffic as normal or malicious (attack) based on various features extracted from the dataset.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Visualization](#visualization)
- [Dependencies](#dependencies)

## Introduction
Intrusion detection is a critical component of cybersecurity. This project leverages machine learning techniques to detect potential threats in network traffic. By training an SVM classifier on the UNSW-NB15 dataset, we aim to achieve high accuracy in distinguishing between normal and attack behaviors.

## Dataset
The UNSW-NB15 dataset is a comprehensive dataset for network intrusion detection research. It contains modern normal activities and contemporary synthesized attack behaviors. The dataset includes:

* Features: 49 attributes representing flow features
* Labels: Binary labels indicating normal or attack traffic
* Attack Categories: Detailed attack types for multi-class classification

**Note:** Ensure that the `UNSW_NB15_training-set.parquet` file is placed in the repository directory.

## Project Structure
### Data Loading and Exploration
* Importing necessary libraries
* Loading the dataset using Pandas
* Displaying initial data insights (head, info, describe)
* Checking for missing values

### Data Visualization
* Plotting the distribution of the target variable (label)
* Visualizing the distribution of attack categories (attack_cat)

### Data Preprocessing
* Encoding categorical variables using LabelEncoder
* Separating features and target variable
* Splitting the data into training and testing sets
* Sampling a subset of the training data for computational efficiency
* Scaling features using StandardScaler

### Model Training
* Initializing the SVM classifier with class_weight='balanced' to handle class imbalance
* Training the model on the sampled data

### Feature Importance
* Calculating feature importance using permutation importance
* Plotting the top 15 most important features

### Model Evaluation
* Making predictions on the test set
* Generating a classification report
* Plotting the confusion matrix
* Plotting the ROC curve and calculating the AUC score

### Model Saving and Deployment
* Saving the trained model, scaler, and label encoders using joblib
* Implementing an intrusion detection function for new data prediction

## Installation

1. Clone the Repository
```bash
git clone https://github.com/your_username/your_repository.git
cd your_repository
```

2. Set Up a Virtual Environment (Optional)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

4. Place the Dataset
Ensure the `UNSW_NB15_training-set.parquet` file is in the project directory.

## Results
### Classification Report
The SVM classifier achieved the following performance on the test set:
```
              precision    recall  f1-score   support

           0       0.99      0.93      0.96     37000
           1       0.83      0.98      0.90     13000

    accuracy                           0.94     50000
   macro avg       0.91      0.96      0.93     50000
weighted avg       0.95      0.94      0.94     50000
```

### Performance Metrics
* Confusion Matrix: Illustrates the correct and incorrect predictions made by the model
* ROC Curve and AUC: The model achieved an AUC score of 0.98, indicating excellent performance in distinguishing between classes

## Visualization
The following visualizations are generated during model training and evaluation:

* Distribution of Target Variable
* Distribution of Attack Categories
* Top 15 Most Important Features
* Confusion Matrix
* ROC Curve

**Note:** The images directory contains all PNG files generated during script execution.

## Dependencies
* Python 3.x
* Required Libraries:
  * pandas
  * numpy
  * matplotlib
  * seaborn
  * scikit-learn
  * joblib
