Network Intrusion Detection System using SVM
Overview
This project implements a Network Intrusion Detection System (NIDS) using Support Vector Machine (SVM) classification. The system is trained on the UNSW-NB15 dataset to detect and classify network attacks.

Features
Data preprocessing and feature engineering
SVM model implementation with balanced class weights
Feature importance analysis
Model evaluation with multiple metrics
Real-time intrusion detection capability
Model persistence for future use
Project Structure
```
├── UNSW_NB15_training-set.parquet    # Training dataset
├── main.py                           # Main implementation file
├── svm_model_balanced.joblib         # Saved SVM model
├── scaler_balanced.joblib            # Saved feature scaler
└── label_encoders_balanced.joblib    # Saved label encoders
```
Dependencies
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
Installation
Clone the repository
```
git clone [repository-url]
```
```
cd [repository-name]
```
Install required packages
```
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```
Usage
Ensure you have the UNSW_NB15_training-set.parquet file in your project directory
Run the main script:

```
python main.py
```
Model Features
The system analyzes various network traffic features including:

Protocol types
Service types
Connection statistics
Flow statistics
Attack categories
Model Performance
The SVM model is evaluated using:

Classification Report
Confusion Matrix
ROC Curve
AUC Score
Real-time Detection
The system includes a detect_intrusions() function for real-time intrusion detection on new network traffic data.

Visualizations
The project includes various visualizations:

Distribution of attack types
Feature importance plots
ROC curves
Confusion matrices
