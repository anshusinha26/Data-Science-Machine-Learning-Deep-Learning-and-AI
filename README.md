# 1 Data Preprocessing Tools

This repository contains code for data preprocessing using Python and scikit-learn.

## Overview

The data preprocessing tools provided in this repository include:

- Handling missing data using `SimpleImputer`
- Encoding categorical data using `OneHotEncoder` and `LabelEncoder`
- Splitting the dataset into training and testing sets using `train_test_split`
- Feature scaling using `StandardScaler`

## Installation

To run this code, make sure you have the following dependencies installed:

- scikit-learn
- pandas
- matplotlib

You can install them using pip:

## Code Usage

1. Clone the repository or download the code files.

2. Ensure that you have the required dependencies installed. You can install them using pip:

3. Modify the code as needed to suit your specific dataset and requirements.

4. Run the code and observe the output.


# 2 Simple Linear Regression

This repository contains code for implementing simple linear regression. It demonstrates how to train a linear regression model using the scikit-learn library and visualize the results.

## Getting Started

To get started with this code, follow the instructions below.

### Prerequisites

Make sure you have the following dependencies installed:

- Python (version 3.6 or higher)
- NumPy
- matplotlib
- pandas
- scikit-learn

## Code Explanation

1. Importing the Libraries

>> The necessary libraries (NumPy, matplotlib, and pandas) are imported to handle data manipulation and visualization.

2. Importing the Dataset

>> The dataset (Salary_Data.csv) is imported using pandas. The independent variable (X) and dependent variable (Y) are extracted from the dataset.

3. Splitting the Dataset

>> The dataset is split into training and test sets using the train_test_split function from scikit-learn. This allows us to evaluate the model's performance on unseen data.

4. Training the Model

>> A linear regression model is created using the LinearRegression class from scikit-learn. The model is trained on the training data using the fit method.

5. Predicting the Test Set Results

>> The trained model is used to make predictions on the test data (X_test), and the predicted values are stored in Y_pred.

6. Visualizing the Results

>> Two plots are created to visualize the results. One plot shows the regression line and the actual data points for the training set, while the other plot shows the regression line and the actual data points for the test set.


# 3 Multiple Linear Regression

This Jupyter Notebook demonstrates how to perform multiple linear regression using the `LinearRegression` class from scikit-learn. It includes the following steps:

1. Importing the necessary libraries.
2. Loading and preparing the dataset.
3. Encoding categorical data.
4. Splitting the dataset into training and test sets.
5. Training the multiple linear regression model on the training set.
6. Predicting the test set results.
7. Printing the predicted and actual values.

## Prerequisites

- Python 3.x
- Jupyter Notebook
- pandas
- scikit-learn
- NumPy
- matplotlib

## Installation

1. Clone the repository or download the notebook file (`Multiple_linear_regression.ipynb`).
2. Install the required libraries using pip:
```bash
pip install pandas scikit-learn matplotlib
```


## Usage

1. Open the notebook in Jupyter Notebook or JupyterLab.
2. Run each cell of the notebook sequentially to see the results.
3. Modify the code as needed for your own dataset or analysis.

## Dataset

The code uses the "50_Startups.csv" dataset, which contains information about 50 startup companies. The dataset includes the following columns:

- R&D Spend: Research and development expenditure
- Administration: Administrative expenditure
- Marketing Spend: Marketing expenditure
- State: State where the startup is located
- Profit: Profit earned by the startup

You can replace the dataset with your own data by providing the appropriate file path or using a different dataset.


# 4 Polynomial Regression

This repository contains a Jupyter Notebook file, Polynomial_regression.ipynb, which demonstrates the implementation of polynomial regression using the scikit-learn library in Python. The notebook can be viewed on Google Colaboratory.

## Overview

In this notebook, we explore polynomial regression, a technique used to model non-linear relationships between variables. We train and compare two regression models: linear regression and polynomial regression, and visualize the results using matplotlib.

## Contents

- Importing the libraries
- Importing the dataset
- Training the Linear Regression model on the whole dataset
- Training the Polynomial Regression model on the whole dataset
- Visualizing the Linear Regression results
- Visualizing the Polynomial Regression results
- Visualizing the Polynomial Regression results (for higher resolution and smoother curve)
- Predicting a new result with Linear Regression
- Predicting a new result with Polynomial Regression

## Usage

To run the code in this notebook, you need to have the following libraries installed:

- numpy
- matplotlib
- pandas
- scikit-learn

You can install the required libraries using pip:
    
```bash
pip install numpy matplotlib pandas scikit-learn
```


# 5 Support Vector Regression (SVR)

This notebook demonstrates the implementation of Support Vector Regression (SVR) using the Scikit-learn library. SVR is a variant of Support Vector Machines (SVM) used for regression tasks.

## Getting Started

### Prerequisites
-    Python 3.x
-    Jupyter Notebook or JupyterLab (recommended)
-    Installation
-    Clone the repository or download the notebook file (Support_vector_regression.ipynb).
-    Install the required libraries using the following command:
-    Copy code
-    pip install numpy pandas matplotlib scikit-learn
-    Run the notebook using Jupyter Notebook or JupyterLab.

## Usage

### The notebook consists of the following main sections:

-    Importing the necessary libraries
-    Importing the dataset
-    Feature Scaling
-    Training the SVR model on the whole dataset
-    Predicting a new result
-    Visualizing the SVR results
-    The code snippets are provided within the notebook, along with explanations for each step.

## Dataset

The notebook uses the "Position_Salaries.csv" dataset, which contains information about different job positions and their corresponding salaries. The dataset is loaded using the pandas library and split into input features (X) and target values (Y).

## Model Training and Evaluation

The SVR model is trained on the entire dataset after scaling the features and target values using the StandardScaler from Scikit-learn. The SVR model is fitted to the scaled data using the fit method.

Predictions are made using the trained model on new input values. The input values are transformed using the scaler, then passed to the predict method of the SVR model. The predictions are inverse transformed using the scaler to obtain the predicted values in their original scale.

The results are visualized using scatter plots and line plots. The scatter plot shows the original data points, while the line plot represents the SVR model's predictions.


# 6 Decision Tree Regression

This repository contains a Jupyter Notebook file (Decision_tree_regression.ipynb) that demonstrates how to perform decision tree regression using the scikit-learn library in Python.

## Description

Decision tree regression is a machine learning algorithm that can be used for both classification and regression tasks. It works by partitioning the input space into regions and predicting the target variable based on the average value of the training examples in each region. This approach creates a tree-like model of decisions and their possible consequences.

In this notebook, we will use decision tree regression to predict salaries based on position levels. The dataset used for training and testing the model is provided in the file Position_Salaries.csv.

## Usage

To run this notebook, follow these steps:

    Install the required libraries (numpy, matplotlib, pandas, and scikit-learn) if they are not already installed.
    Download the Position_Salaries.csv file and place it in the same directory as this notebook.
    Open the notebook using Jupyter Notebook or Google Colaboratory.
    Execute each cell in the notebook sequentially to see the output and visualize the results.
    Note: Make sure to have a working Python environment with the necessary dependencies installed.

## Contents

The notebook consists of the following sections:

    Importing the libraries: This section imports the required libraries for the analysis.
    Importing the dataset: The dataset is loaded into the notebook and split into input features (X) and target variable (Y).
    Training the Decision Tree Regression model: A DecisionTreeRegressor model is trained on the entire dataset.
    Predicting a new result: The trained model is used to predict the salary for a new position level.
    Visualizing the Decision Tree Regression results: The results are plotted on a graph to visualize the regression line.


# 7 Random Forest Regression

This code demonstrates the implementation of Random Forest Regression using scikit-learn. It uses a dataset called "Position_Salaries.csv" to predict the salary based on the position level.

## Getting Started

To run the code, you will need to have Python installed on your machine along with the following libraries:
- numpy
- matplotlib
- pandas
- scikit-learn

You can install the required libraries using pip:

```bash
pip install numpy matplotlib pandas scikit-learn
```


## Files

- `Random_forest_regression.ipynb`: Jupyter Notebook file containing the code.

## Usage

1. Clone the repository or download the `Random_forest_regression.ipynb` file.
2. Open the Jupyter Notebook in your preferred environment.
3. Make sure you have the dataset file "Position_Salaries.csv" in the same directory as the notebook.
4. Run the code cells in the notebook sequentially to execute the code.

## Dataset

The dataset used in this code is called "Position_Salaries.csv". It contains two columns: "Level" (representing the position level) and "Salary" (the corresponding salary). The code reads this dataset and splits it into input features (X) and target variable (Y).

## Model Training

The Random Forest Regression model is trained on the entire dataset using the RandomForestRegressor class from scikit-learn. The number of decision trees (n_estimators) is set to 10, and the random_state parameter is set to 1 for result reproducibility.

## Prediction

The code predicts the salary for a new position level (6.5) using the trained Random Forest Regression model.

## Visualization

The code visualizes the Random Forest Regression results with a scatter plot of the actual data points and a continuous line representing the predicted values. The plot provides a higher resolution view of the regression curve.





