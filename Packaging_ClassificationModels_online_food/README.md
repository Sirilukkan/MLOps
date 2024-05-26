# Overview and Goal of This Project

The main goal of this project is to package a machine learning model using Python and essential files to ensure that the code or machine learning predictions can run in any environment with the necessary dependencies. This project aims to predict customer feedback to improve service quality based on demographic factors and food ordering behaviors.

# Dataset

### Food Online Dataset

This dataset can be obtained from [Kaggle](https://www.kaggle.com/datasets/sudarshan24byte/online-food-dataset).

### Data Description

The dataset contains information collected from an online food ordering platform over a period of time. It encompasses various attributes related to occupation, family size, feedback, etc.

### Data Attributes

| Attribute                  | Description                                                   |
|----------------------------|---------------------------------------------------------------|
| **Demographic Information**|                                                               |
| Age                        | Age of the customer.                                          |
| Gender                     | Gender of the customer.                                       |
| Marital Status             | Marital status of the customer.                               |
| Occupation                 | Occupation of the customer.                                   |
| Monthly Income             | Monthly income of the customer.                               |
| Educational Qualifications | Educational qualifications of the customer.                   |
| Family Size                | Number of individuals in the customer's family.               |
| **Location Information**   |                                                               |
| Latitude                   | Latitude of the customer's location.                          |
| Longitude                  | Longitude of the customer's location.                         |
| Pin Code                   | Pin code of the customer's location.                          |
| **Order Details**          |                                                               |
| Output                     | Current status of the order (e.g., pending, confirmed, delivered). |
| Feedback                   | Feedback provided by the customer after receiving the order.  |

# Machine Learning Models

I selected four machine learning models and performed grid search to find the best model with the best hyperparameters. The models selected are:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost

# Description of Each Folder

### Experiments

Contains Python code step-by-step for cleaning, preprocessing, machine learning model building, and prediction in Jupyter Notebooks (.ipynb). Experimenting with code in Jupyter Notebooks is easier as it allows for interactive, flexible, and visually rich environment, making it easier to see each execution and fix issues before writing the final code.

### packaging-ml-model

This is the main folder where the ML model is packaged. Important files needed for packaging ML models are described in the README.md file inside this folder.

# Acknowledgement

I learned how to package ML models from the Udemy course [MLOps Bootcamp: Mastering AI Operations for Success - AIOps](https://www.udemy.com/course/mlops-bootcamp-mastering-ai-operations-for-success-aiops/?couponCode=OF52424).


