# Hyperparameter Tuning and Optimization using Random Forest

## Project Overview

This project demonstrates hyperparameter tuning and optimization of a machine learning model using the Titanic Survival Prediction dataset. The objective is to improve model performance by selecting the best combination of hyperparameters through Grid Search Cross Validation.

## Objectives

* Understand machine learning hyperparameters
* Apply Grid Search Cross Validation
* Optimize Random Forest Classifier performance
* Compare baseline and tuned models
* Evaluate model performance using classification metrics

## Dataset

The Titanic dataset contains passenger information and survival status.

### Features

* Passenger Class (Pclass)
* Sex
* Age
* Number of Siblings/Spouses Aboard (SibSp)
* Number of Parents/Children Aboard (Parch)
* Fare
* Embarked

### Target Variable

* Survived

## Data Preprocessing

The following preprocessing steps were performed:

1. Removed unnecessary columns:

   * PassengerId
   * Name
   * Ticket
   * Cabin

2. Handled missing values:

   * Age filled using median value
   * Embarked filled using mode value

3. Encoded categorical variables using one-hot encoding.

## Model Used

Random Forest Classifier

## Hyperparameter Tuning

Grid Search Cross Validation (CV = 5) was used to identify the optimal hyperparameter combination.

### Parameters Tested

* n_estimators = [50, 100, 200]
* max_depth = [5, 10, 20]
* min_samples_split = [2, 5, 10]
* min_samples_leaf = [1, 2, 4]

## Results

### Baseline Model Accuracy

82.12%

### Best Hyperparameters

```python
{
    'max_depth': 5,
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'n_estimators': 100
}
```

### Best Cross Validation Score

83.56%

### Tuned Model Accuracy

81.56%

### Classification Report

| Class            | Precision | Recall | F1-Score |
| ---------------- | --------- | ------ | -------- |
| Not Survived (0) | 0.81      | 0.90   | 0.85     |
| Survived (1)     | 0.84      | 0.69   | 0.76     |

Overall Accuracy: 82%

Weighted F1-Score: 81%

## Conclusion

Hyperparameter tuning was successfully performed using Grid Search Cross Validation. The best hyperparameter combination achieved a cross-validation score of 83.56%. The project demonstrates how systematic parameter optimization can improve model selection and provide better generalization performance.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* GridSearchCV

## Author

Varunjitha M

