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
This section explains the process of training the model. All models and data processing steps are performed using a
random_state of 42. The Data is splitted in training and testing with a test size of (0.2). All models were loaded from
the sklearn module.
## Preprocessing
The project includes visualization of data in the preprocessing step. For further information Check the pictures in 
/figures. During this step the whole dataset is checked for duplicates. Furthermore, the values of each feature is 
checked for being in the correct range and if correct dtype and for missing values. Instances of features with missing
values are deleted from the dataset. The processed dataset is then saved in /data as a pickeld object.
## ML
### Logistic Regression
This model is trained using the sklearn predefined hyper-parameters. In order to process the data a standard scaler is
implemented to standardize the data. 
A cross-validation method with 6 folds and a 
shuffel parameter set to "True" is performed to gather the best possible Regression for this dataset.
### k-Nearest Neighbors
For this model there is hyper-parameter tuning implemented. The following values were set for a GridsearchCV hyper-parameter
tuning:
1. **n_neighbors**: [3, 4, 5, 6, 7, 8]
2. **metric**: ['minkowski', 'manhattan']
This model is again trained using a 6 times folded cross-validation with shuffel: "True"
GridSearch defined the following hyper-parameter as optimal setting:

### Random forest
GridSearchCV is also implemented for this model while settings for KFold and shuffel stay the same as in the other models.
Parameters which are tuned:
1. **n_estimators**: [250, 300, 400]
2. **max_features**: ['sqrt', 'log2']
3. **max_depth**: [2, 3, 4, 5]
4. **criterion**: ['gini', 'entropy']

# Results
The performance of the models is evaluated using multiple metrics.
The best performance to predict the data was the Logistic Regression model with the following metrics:
- Accuracy: 85%
- F1 Score: 0.82
- Confusion Matrix

<p align="center">
    <img src="https://github.com/sandro-roth/Heart_Disease_Prediction/blob/main/main/Results/Logistic_Regression/confusion_matrix.png?raw=true" width="350">
</p>

# Contributors
- [Sandro Roth]