# 🟥 Regression 

## 🔹 Data Preprocessing Tools

This repository contains code for data preprocessing using Python and scikit-learn.

### Overview

The data preprocessing tools provided in this repository include:

- Handling missing data using `SimpleImputer`
- Encoding categorical data using `OneHotEncoder` and `LabelEncoder`
- Splitting the dataset into training and testing sets using `train_test_split`
- Feature scaling using `StandardScaler`

### Installation

To run this code, make sure you have the following dependencies installed:

- scikit-learn
- pandas
- matplotlib

You can install them using pip:

### Code Usage

1. Clone the repository or download the code files.

2. Ensure that you have the required dependencies installed. You can install them using pip:

3. Modify the code as needed to suit your specific dataset and requirements.

4. Run the code and observe the output.


## 🔹 Simple Linear Regression

This repository contains code for implementing simple linear regression. It demonstrates how to train a linear regression model using the scikit-learn library and visualize the results.

### Getting Started

To get started with this code, follow the instructions below.

#### Prerequisites

Make sure you have the following dependencies installed:

- Python (version 3.6 or higher)
- NumPy
- matplotlib
- pandas
- scikit-learn

### Code Explanation

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

## 🔹 Multiple Linear Regression

This Jupyter Notebook demonstrates how to perform multiple linear regression using the `LinearRegression` class from scikit-learn. It includes the following steps:

1. Importing the necessary libraries.
2. Loading and preparing the dataset.
3. Encoding categorical data.
4. Splitting the dataset into training and test sets.
5. Training the multiple linear regression model on the training set.
6. Predicting the test set results.
7. Printing the predicted and actual values.

### Prerequisites

- Python 3.x
- Jupyter Notebook
- pandas
- scikit-learn
- NumPy
- matplotlib

### Installation

1. Clone the repository or download the notebook file (`Multiple_linear_regression.ipynb`).
2. Install the required libraries using pip:
```bash
    pip install pandas scikit-learn matplotlib
```


### Usage

1. Open the notebook in Jupyter Notebook or JupyterLab.
2. Run each cell of the notebook sequentially to see the results.
3. Modify the code as needed for your own dataset or analysis.

### Dataset

The code uses the "50_Startups.csv" dataset, which contains information about 50 startup companies. The dataset includes the following columns:

- R&D Spend: Research and development expenditure
- Administration: Administrative expenditure
- Marketing Spend: Marketing expenditure
- State: State where the startup is located
- Profit: Profit earned by the startup

You can replace the dataset with your own data by providing the appropriate file path or using a different dataset.

## 🔹 Polynomial Regression

This repository contains a Jupyter Notebook file, Polynomial_regression.ipynb, which demonstrates the implementation of polynomial regression using the scikit-learn library in Python. The notebook can be viewed on Google Colaboratory.

### Overview

In this notebook, we explore polynomial regression, a technique used to model non-linear relationships between variables. We train and compare two regression models: linear regression and polynomial regression, and visualize the results using matplotlib.

### Contents

- Importing the libraries
- Importing the dataset
- Training the Linear Regression model on the whole dataset
- Training the Polynomial Regression model on the whole dataset
- Visualizing the Linear Regression results
- Visualizing the Polynomial Regression results
- Visualizing the Polynomial Regression results (for higher resolution and smoother curve)
- Predicting a new result with Linear Regression
- Predicting a new result with Polynomial Regression

### Usage

To run the code in this notebook, you need to have the following libraries installed:

- numpy
- matplotlib
- pandas
- scikit-learn

You can install the required libraries using pip:
    
```bash
    pip install numpy matplotlib pandas scikit-learn
```

## 🔹 Support Vector Regression (SVR)

This notebook demonstrates the implementation of Support Vector Regression (SVR) using the Scikit-learn library. SVR is a variant of Support Vector Machines (SVM) used for regression tasks.

### Getting Started

#### Prerequisites
-    Python 3.x
-    Jupyter Notebook or JupyterLab (recommended)
-    Installation
-    Clone the repository or download the notebook file (Support_vector_regression.ipynb).
-    Install the required libraries using the following command:
-    Copy code
-    pip install numpy pandas matplotlib scikit-learn
-    Run the notebook using Jupyter Notebook or JupyterLab.

### Usage

#### The notebook consists of the following main sections:

-    Importing the necessary libraries
-    Importing the dataset
-    Feature Scaling
-    Training the SVR model on the whole dataset
-    Predicting a new result
-    Visualizing the SVR results
-    The code snippets are provided within the notebook, along with explanations for each step.

### Dataset

The notebook uses the "Position_Salaries.csv" dataset, which contains information about different job positions and their corresponding salaries. The dataset is loaded using the pandas library and split into input features (X) and target values (Y).

### Model Training and Evaluation

The SVR model is trained on the entire dataset after scaling the features and target values using the StandardScaler from Scikit-learn. The SVR model is fitted to the scaled data using the fit method.

Predictions are made using the trained model on new input values. The input values are transformed using the scaler, then passed to the predict method of the SVR model. The predictions are inverse transformed using the scaler to obtain the predicted values in their original scale.

The results are visualized using scatter plots and line plots. The scatter plot shows the original data points, while the line plot represents the SVR model's predictions.

## 🔹 Decision Tree Regression

This repository contains a Jupyter Notebook file (Decision_tree_regression.ipynb) that demonstrates how to perform decision tree regression using the scikit-learn library in Python.

### Description

Decision tree regression is a machine learning algorithm that can be used for both classification and regression tasks. It works by partitioning the input space into regions and predicting the target variable based on the average value of the training examples in each region. This approach creates a tree-like model of decisions and their possible consequences.

In this notebook, we will use decision tree regression to predict salaries based on position levels. The dataset used for training and testing the model is provided in the file Position_Salaries.csv.

### Usage

To run this notebook, follow these steps:

    Install the required libraries (numpy, matplotlib, pandas, and scikit-learn) if they are not already installed.
    Download the Position_Salaries.csv file and place it in the same directory as this notebook.
    Open the notebook using Jupyter Notebook or Google Colaboratory.
    Execute each cell in the notebook sequentially to see the output and visualize the results.
    Note: Make sure to have a working Python environment with the necessary dependencies installed.

### Contents

The notebook consists of the following sections:

    Importing the libraries: This section imports the required libraries for the analysis.
    Importing the dataset: The dataset is loaded into the notebook and split into input features (X) and target variable (Y).
    Training the Decision Tree Regression model: A DecisionTreeRegressor model is trained on the entire dataset.
    Predicting a new result: The trained model is used to predict the salary for a new position level.
    Visualizing the Decision Tree Regression results: The results are plotted on a graph to visualize the regression line.

## 🔹 Random Forest Regression

This code demonstrates the implementation of Random Forest Regression using scikit-learn. It uses a dataset called "Position_Salaries.csv" to predict the salary based on the position level.

### Getting Started

To run the code, you will need to have Python installed on your machine along with the following libraries:
- numpy
- matplotlib
- pandas
- scikit-learn

You can install the required libraries using pip:

```bash
    pip install numpy matplotlib pandas scikit-learn
```


### Files

- `Random_forest_regression.ipynb`: Jupyter Notebook file containing the code.

### Usage

1. Clone the repository or download the `Random_forest_regression.ipynb` file.
2. Open the Jupyter Notebook in your preferred environment.
3. Make sure you have the dataset file "Position_Salaries.csv" in the same directory as the notebook.
4. Run the code cells in the notebook sequentially to execute the code.

### Dataset

The dataset used in this code is called "Position_Salaries.csv". It contains two columns: "Level" (representing the position level) and "Salary" (the corresponding salary). The code reads this dataset and splits it into input features (X) and target variable (Y).

### Model Training

The Random Forest Regression model is trained on the entire dataset using the RandomForestRegressor class from scikit-learn. The number of decision trees (n_estimators) is set to 10, and the random_state parameter is set to 1 for result reproducibility.

### Prediction

The code predicts the salary for a new position level (6.5) using the trained Random Forest Regression model.

### Visualization

The code visualizes the Random Forest Regression results with a scatter plot of the actual data points and a continuous line representing the predicted values. The plot provides a higher resolution view of the regression curve.


# 🟧 Classification

## 🔹 Logistic Regression

This code performs logistic regression on a dataset to predict whether a customer will purchase a product based on their age and estimated salary. It uses the scikit-learn library to train a logistic regression model and evaluate its performance.

### Dependencies

Make sure you have the following dependencies installed:

- NumPy
- Matplotlib
- Pandas
- scikit-learn

You can install them using pip:
```bash
    pip install numpy matplotlib pandas scikit-learn
```

### Dataset

The code assumes that you have a CSV file named "Social_Network_Ads.csv" containing the dataset. You can replace the file path in the code with the correct path to your dataset.

### Running the Code

- Make sure you have the dataset file available and the dependencies installed.
- Run the code in a Python environment such as Jupyter Notebook or a Python script.
- The code will load the dataset, split it into training and test sets, perform feature scaling, and train a logistic regression model.
- It will then make predictions on the test set and display the results, including the confusion matrix, accuracy score, and R-squared score.
- Additionally, it will visualize the training and test set results using scatter plots and decision boundaries.

## 🔹 K-Nearest Neighbors (K-NN) Classifier

This code demonstrates the implementation of the K-Nearest Neighbors (K-NN) algorithm using scikit-learn library. It predicts whether a user would purchase a product based on their age and estimated salary.

### Dataset
The dataset used in this code is "Social_Network_Ads.csv". It contains information about users, including their age, estimated salary, and whether they purchased a product. The goal is to build a K-NN classifier to predict if a user will purchase a product based on age and estimated salary.

### Code Overview
The code performs the following steps:

1. Importing the necessary libraries: NumPy, pandas, Matplotlib, and scikit-learn.

2. Importing the dataset: The dataset is loaded from the "Social_Network_Ads.csv" file and split into features (X) and target (Y).

3. Splitting the dataset: The data is split into training and test sets using the `train_test_split` function from scikit-learn.

4. Feature Scaling: The features in the dataset are scaled using the `StandardScaler` from scikit-learn to ensure that they have the same scale.

5. Training the K-NN model: A K-NN classifier is created using the `KNeighborsClassifier` class from scikit-learn and trained on the training set.

6. Predicting a new result: The trained model is used to predict if a user will purchase a product given their age and estimated salary.

7. Predicting the Test set results: The model is used to predict the purchases for the test set, and the predicted values are compared to the actual values.

8. Making the Confusion Matrix: A confusion matrix is computed using the `confusion_matrix` function from scikit-learn to evaluate the model's performance.

9. Visualizing the Training set and Test set results: The training set and test set results are visualized using a scatter plot to show the predicted and actual values. The decision boundary is also displayed.

### Instructions
1. Ensure that you have the required libraries installed: NumPy, pandas, Matplotlib, and scikit-learn.

2. Download the "Social_Network_Ads.csv" dataset and place it in the same directory as the code.

3. Run the code to train the K-NN classifier and visualize the results.

Feel free to modify the code to experiment with different parameters or apply it to your own datasets.

## 🔹 Support Vector Machine (SVM)

This repository contains a Jupyter Notebook (`Support_vector_machine.ipynb`) that demonstrates how to use a Support Vector Machine (SVM) classifier for classification tasks. It uses the scikit-learn library in Python.

Support Vector Machine is a powerful supervised learning algorithm used for both classification and regression tasks. It constructs a hyperplane or set of hyperplanes in a high-dimensional feature space to separate different classes. In this notebook, we use an SVM classifier with an RBF kernel.

### Getting Started

To run the notebook and execute the code, follow these steps:

1. Clone the repository to your local machine or download the `Support_vector_machine.ipynb` file directly.

2. Install the required dependencies by running the following command:

```bash
    pip install scikit-learn matplotlib pandas
```

3. Launch Jupyter Notebook by running the command:

```bash
    jupyter notebook
```

4. Open the `Support_vector_machine.ipynb` notebook in your Jupyter Notebook environment.

5. Execute the code cells in the notebook sequentially to see the SVM classifier in action.

### Contents

The notebook covers the following topics:

1. Importing the necessary libraries.
2. Importing the dataset (`Social_Network_Ads.csv`).
3. Splitting the dataset into the training set and test set.
4. Feature scaling using the StandardScaler.
5. Training the SVM model on the training set.
6. Predicting new results and evaluating the model's performance.
7. Visualizing the training set and test set results.

### Dataset

The dataset used in this example (`Social_Network_Ads.csv`) contains information about users' age, estimated salary, and whether they purchased a particular product or not. The goal is to train an SVM model to predict whether a user will purchase the product based on their age and estimated salary.

## 🔹 Kernel SVM

This repository contains a Jupyter Notebook file (`Kernel_svm.ipynb`) that demonstrates the implementation of a kernel support vector machine (SVM) model for classification. The code uses the scikit-learn library in Python.

### Dataset

The code uses the "Social_Network_Ads.csv" dataset, which is included in the repository. The dataset contains information about users' age, estimated salary, and whether they purchased a product (target variable).

### Getting Started

1. Clone the repository:

```bash 
   git clone https://github.com/your-username/kernel-svm.git
   cd kernel-svm
```

2. Install the required dependencies. You can use pip to install the necessary packages:

```bash
    pip install pandas matplotlib scikit-learn
```

3. Run the Jupyter Notebook:

```bash
    jupyter notebook Kernel_svm.ipynb
```

This will open the notebook in your browser.

4. Follow the instructions in the notebook to execute each code cell and observe the results.

### Results

The notebook contains code for various steps, including data preprocessing, model training, prediction, evaluation, and visualization. It demonstrates the following:

- Importing and exploring the dataset
- Splitting the dataset into training and test sets
- Feature scaling
- Training a kernel SVM model on the training set
- Predicting new results using the trained model
- Evaluating the model's performance using a confusion matrix and accuracy score
- Visualizing the results on both the training and test sets

## 🔹 Naive Bayes Classifier

This repository contains an example implementation of the Naive Bayes classifier using scikit-learn library in Python. The code demonstrates how to train a Naive Bayes model on a dataset, make predictions, and visualize the results.

### Dependencies

Make sure you have the following dependencies installed:

- Python (3.x recommended)
- NumPy
- matplotlib
- pandas
- scikit-learn

You can install the required dependencies using pip:

```bash
    pip install numpy matplotlib pandas scikit-learn
```

### Dataset

The code uses the "Social_Network_Ads.csv" dataset, which contains information about users in a social network. The dataset has the following columns:

- `Age`: Age of the user
- `EstimatedSalary`: Estimated salary of the user
- `Purchased`: Whether the user purchased a product (1 if purchased, 0 otherwise)

### Usage

1. Clone this repository or download the "Naive_bayes.ipynb" file.

2. Open the notebook in Jupyter Notebook or any other compatible environment.

3. Run the notebook cell by cell to see the step-by-step process of training the Naive Bayes model, making predictions, and visualizing the results.

Note: Make sure to have the "Social_Network_Ads.csv" file in the same directory as the notebook.

## 🔹 Decision Tree Classification

This repository contains an example implementation of the Decision Tree Classification algorithm using scikit-learn library in Python. The code demonstrates how to train a Decision Tree model on a dataset, make predictions, and visualize the results.

### Dependencies

Make sure you have the following dependencies installed:

- Python (3.x recommended)
- NumPy
- matplotlib
- pandas
- scikit-learn

You can install the required dependencies using pip:

```bash
pip install numpy matplotlib pandas scikit-learn
```


### Dataset

The code uses the "Social_Network_Ads.csv" dataset, which contains information about users in a social network. The dataset has the following columns:

- `Age`: Age of the user
- `EstimatedSalary`: Estimated salary of the user
- `Purchased`: Whether the user purchased a product (1 if purchased, 0 otherwise)

### Usage

1. Clone this repository or download the "Decision_tree_classification.ipynb" file.

2. Open the notebook in Jupyter Notebook or any other compatible environment.

3. Run the notebook cell by cell to see the step-by-step process of training the Decision Tree Classification model, making predictions, and visualizing the results.

Note: Make sure to have the "Social_Network_Ads.csv" file in the same directory as the notebook.

## 🔹 Random Forest Classification

This repository contains an example of training a Random Forest Classifier model for classification using the scikit-learn library in Python.

### Dataset

The dataset used in this example is "Social_Network_Ads.csv". It contains information about users in a social network, including their age and estimated salary, as well as whether or not they purchased a product. The goal is to predict whether a user will purchase a product based on their age and estimated salary.

### Dependencies

- Python (version 3.x)
- scikit-learn (version 0.24.2)
- NumPy (version 1.20.3)
- Matplotlib (version 3.4.2)
- pandas (version 1.3.0)

### Instructions

1. Clone the repository or download the "Random_forest_classification.ipynb" file.

2. Install the required dependencies mentioned above, preferably using a virtual environment.

3. Run the Jupyter notebook file "Random_forest_classification.ipynb" using Jupyter Notebook or any compatible IDE.

4. The notebook will guide you through the following steps:

   - Importing the necessary libraries.
   - Loading and preprocessing the dataset.
   - Splitting the data into training and test sets.
   - Scaling the features.
   - Training the Random Forest Classifier model.
   - Making predictions on new data.
   - Evaluating the model's performance using a confusion matrix and accuracy score.
   - Visualizing the results on the training and test sets.

Note: The code assumes that the dataset file "Social_Network_Ads.csv" is present in the same directory as the notebook file.


# 🟨 Clustering

## 🔹 K-Means Clustering

This repository contains a Jupyter Notebook (`K_means_clustering.ipynb`) that demonstrates the application of K-means clustering algorithm on a dataset of mall customers.

### Contents

- `K_means_clustering.ipynb`: Jupyter Notebook containing the code and analysis.
- `Mall_Customers.csv`: Dataset file (CSV format) containing customer information.

### Dataset

The dataset used in this project is called "Mall_Customers.csv." It provides information about customers of a mall, including their annual income and spending score. The goal is to segment the customers into different clusters based on these attributes.

### Prerequisites

To run the code in the Jupyter Notebook, the following libraries are required:

- NumPy
- Pandas
- Matplotlib
- Scikit-learn

You can install these libraries using pip or any other package manager.

### How to Use

1. Clone the repository or download the files.
2. Make sure you have the required libraries installed.
3. Open the `K_means_clustering.ipynb` notebook in Jupyter Notebook or any compatible environment.
4. Run the notebook cells to execute the code step-by-step.
5. The notebook demonstrates the following:
   - Importing the dataset
   - Using the elbow method to determine the optimal number of clusters
   - Training the K-means model on the dataset
   - Visualizing the clusters

### Results

The main result of this project is the visualization of customer clusters using the K-means algorithm. The scatter plot shows the different clusters identified and the centroids of each cluster.

The analysis can provide insights into customer segmentation, helping businesses understand and target different customer groups based on their income and spending behavior.
