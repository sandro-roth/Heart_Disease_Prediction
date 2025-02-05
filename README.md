# Heart Disease Prediction

## Project Overview

This Project aims to develop a model for predicting heart disease. It uses different Classification Algorithms from
sklearn. For the current state of the Project 3 Models are applied (Logistic Regression, k-Nearest Neighbors and
Random Forest). The origin of the Data is the UCI Heart Disease Dataset [https://archive.ics.uci.edu/dataset/45/heart+disease]

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model Training](#model-training)
- [Results](#results)
- [Contributors](#contributors)
- [License](#license)

# Installation

Follow these steps to set up the project on your local machine:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/sandro-roth/Heart_Disease_Prediction.git
    cd Heart_Disease_Prediction
   
2. **Create the virtual environment**
    ```bash
   python3 -m venv venv
   source venv/bin/activate #On Windows: venv/Scripts/activate
   
3. **Install dependencies**
    ```bash
   pip install -r requirements.txt
   
# Usage
In order to run the full project the main file has to be run your virtual environment. At the current stage the models
are not saved and have to be retrained every time. The following commands in the main.py file make sure that pre-processing
and traing/testing of the models is happening. One can comment the first "prepare()" function after running the code
once. Since the preprocessed Data is pickled.
1. **Check the end of main.py**
    ```python
   if __name__ == '__main__':
       prepare()
       learn()
    ```

2. **Now run the code with the following command**
    ```bash
   python main.py
    ```
   
# Data
The dataset used in this project comes from the UCI Machine Learning Repository and is open source. It contains the
following characteristics:
- Number of features: 13
- Number of instances: 303
- Target variable: Presence of heart disease (Values 0 - 4)

The target variable was converted into a binary classification problem for this project:
- 0: Indicates a normal status, meaning no heart disease.
- 1: Indicates the presence of heart disease.

# Model Training
upcoming

# Results
The performance of the models is evaluated using multiple metrics.
The best performance to predict the data was the Logistic Regression model with the following metrics:
- Accuracy: 85%
- F1 Score: 0.82
- Confusion Matrix

<p align="center">
    <img src="https://github.com/sandro-roth/Heart_Disease_Prediction/blob/main/main/Results/Logistic_Regression/confusion_matrix.png?raw=true" width="500">
</p>

# 
