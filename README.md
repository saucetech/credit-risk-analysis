# credit-risk-analysis

## Overview of the Analysis

This project aims to employ machine learning techniques to create a logistic regression model that can identify the creditworthiness of borrowers based on historical lending activity from a peer-to-peer lending services company. Using both original data and resampled data, the project acquires a count of the target classes, trains a logistic regression classifier, calculated teh balanced accuracy score, and generates a confusion matrix as well as a classification report.

The data used in this project includes lending data from a csv file that includes loan size, interest rate, borrower income, debt to income ratio, number of accounts, derogatory marks, total debt, and loan status. The training model uses the loan status (0 for healthy and 1 for high risk of defaulting) as the target, and all other columns as the features. The model is then used to predict the loan status.

The machine learning process includes model-fit-predict. In the first step, the data is imported into the notebook, and the features and targets are set. Then, the data is split into training and testing datasets using train_test_split. A logistic regression model is created via LogisticRegression, and the model is fit using the training data. Lastly, the model is then used to predict the testing data by using the testing feature data and the fitted model. In order to evaluate the model's performance, a balanced accuracy score, confusion matrix, and classification report are created for the model. These steps are repeated using resampled data via RandomOverSampler.

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  - Balanced Accuracy Score: 0.952
  - Precision 0: 1.00
  - Precision 1: 0.85
  - Recall 0: 0.91
  - Recall 1: 0.99
  - F1 0: 1.00
  - F1 1: 0.88



* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.
  - Balanced Accuracy Score: 0.994
  - Precision 0: 1.00
  - Precision 1: 0.84
  - Recall 0: 0.99
  - Recall 1: 0.99
  - F1 0: 1.00
  - F1 1: 0.91

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

Credit risk classification can be problematic because healthy loans outnumber risky loans, making it an imbalanced class. It should be easier to predict healthy loans (0) because there is a greater sample size. Therefore, resampling of the data should provide a greater accuracy for high risk loans, thus providing greater overall accuracy for the entire model. The results reflect this hypothesis as the model with the resampled data had a higher balanced accuracy score of 0.994 versus 0.952 of the original data. The model with resampled data showed higher performance in recall for healthy loans and F1 score for high risk loans. Due to a higher accuracy score, I would su


## Technologies

This project leverages Python 3.7 with the following packages:
numpy, pandas, pathlib, sklearn.metrics, imblearn.metrics

## Installation

Before running this notebook, ensure that you have sklearn and imblearn installed in your environment.

You can install sklearn via the following code:
```
pip install -U scikit-learn
```

You can install imbalanced learn via the following code:
```
conda install -c conda-forge imbalanced-learn
```


## Usage

In order to use view this project, simply clone the repo and run the notebook to view the analysis.

## Contributors

Brought to you by Austin Do. Email: austindotech@gmail.com

## License

MIT