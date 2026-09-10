# Module 2 - Titanic Analytics and Machine Learning

## Overview

This module performs exploratory data analysis, statistical analysis, visualization, classification, class imbalance analysis, Random Forest tuning, and multivariate linear regression using the classic Titanic dataset.

## Dataset Loading

The Titanic dataset is loaded once using `seaborn.load_dataset("titanic")`.

The raw dataset is immediately saved as `titanic.csv` inside this module. This committed CSV provides an offline fallback and prevents the modeling stage from requiring another dataset download.

The modeling stage uses `cleaned_titanic.csv`.

## Exploratory Data Analysis

The initial dataset contains 891 rows and 15 columns.

Missing values were evaluated for every affected column.

The `deck` column had 77.22% missing values and was removed because the missing proportion was above 30%.

The `age` column had 19.87% missing values and was handled using median imputation because its missing percentage was between 5% and 30%.

The `embarked` column had 0.22% missing values and affected rows were removed because the missing percentage was below 5%.

The `embark_town` column had 0.22% missing values and affected rows were removed for the same reason.

The cleaned dataset contains 889 rows with no remaining missing values.

## Outlier Analysis

The 1.5 IQR rule identified:

- Age outliers: 65
- Fare outliers: 114

Outliers were reported but not automatically removed because they represent potentially meaningful passenger observations.

## Fare Distribution

Fare mean: 32.0967

Fare median: 14.4542

Fare mode: 8.05

Because mean > median > mode, the fare distribution is right-skewed.

## Survival Analysis

Female survival rate: 74.04%

Male survival rate: 18.89%

First-class survival rate: 62.62%

Second-class survival rate: 47.28%

Third-class survival rate: 24.24%

Sex and passenger-class combinations were calculated using boolean masks with `&`.

Female first-class survival rate: 96.74%

Female second-class survival rate: 92.11%

Female third-class survival rate: 50.00%

Male first-class survival rate: 36.89%

Male second-class survival rate: 15.74%

Male third-class survival rate: 13.54%

The results show that both sex and passenger class are strongly associated with survival outcomes.

## Correlation Analysis

The correlation matrix contains exactly the required six variables:

- survived
- pclass
- age
- sibsp
- parch
- fare

The two strongest off-diagonal correlations by absolute value are:

1. pclass and fare: -0.5482
2. sibsp and parch: 0.4145

Passenger class and fare have a moderate negative correlation because lower numerical passenger-class values represent higher classes with generally higher fares.

SibSp and Parch have a moderate positive correlation because passengers travelling with siblings or spouses were also more likely to travel with parents or children aboard.

## Multivariate Visualizations

### Survival by Sex and Passenger Class

Survival varies substantially across both sex and passenger class. Female passengers have substantially higher survival rates, and higher passenger classes generally have better survival outcomes.

### Age, Fare and Survival

The scatter plot shows the combined relationship between age, fare and survival. Surviving passengers are more concentrated among higher-fare observations, while age shows substantial variation.

### Fare by Passenger Class and Survival

Fare distributions differ considerably across passenger classes. Higher passenger classes generally have higher fares, while survival also varies within each class.

### Passenger Count by Class and Sex

Third class contains the largest number of passengers. The distribution of male and female passengers also differs across passenger classes.

## Standardization

Age and fare were standardized using `StandardScaler` on the full cleaned dataset for exploratory analysis.

After standardization, both variables have means approximately equal to zero and standard deviations approximately equal to one.

This exploratory standardization was not used as input to the modeling stage.

## Classification Pipeline

The classification target is `survived`.

Features used:

- pclass
- age
- sibsp
- parch
- fare
- sex
- embarked

An 80/20 stratified train-test split was used with random state 42.

Stratification preserves a similar survival-class distribution between the training and test sets.

Preprocessing is implemented using a `ColumnTransformer` and `Pipeline`.

Numeric variables use median imputation followed by `StandardScaler`.

Categorical variables use most-frequent imputation followed by one-hot encoding.

The preprocessing stages are fitted only on the training data to prevent data leakage.

## Classification Models

Three classifiers were trained using the identical train-test split:

- Logistic Regression
- Decision Tree
- Random Forest

Results:

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8090 | 0.7833 | 0.6912 | 0.7344 | 0.8610 |
| Decision Tree | 0.7640 | 0.7600 | 0.5588 | 0.6441 | 0.8374 |
| Random Forest | 0.8034 | 0.7619 | 0.7059 | 0.7328 | 0.8237 |

Logistic Regression provides the strongest overall balance among the three required classifiers and has the highest AUC.

## Class Imbalance

The target distribution is:

- Not Survived: 549 observations, 61.75%
- Survived: 340 observations, 38.25%

Three Logistic Regression strategies were compared:

| Strategy | Precision | Recall | F1 |
|---|---:|---:|---:|
| Baseline | 0.7833 | 0.6912 | 0.7344 |
| Class Weight Balanced | 0.7183 | 0.7500 | 0.7338 |
| SMOTE | 0.7353 | 0.7353 | 0.7353 |

SMOTE produced the highest F1 score among the three imbalance strategies, although the improvement over the baseline was small.

SMOTE was applied only within the training pipeline so that synthetic observations were never generated from the test set.

## Random Forest Grid Search

GridSearchCV was used to tune:

- `n_estimators`
- `max_depth`
- `max_features`

The best parameters were:

```text
max_depth = 5
max_features = sqrt
n_estimators = 200
```

Best cross-validation F1: 0.7408

Out-of-bag score: 0.8214

The tuned Random Forest achieved:

- Accuracy: 0.8315
- Precision: 0.8654
- Recall: 0.6618
- F1: 0.7500
- AUC: 0.8389

## Regression

Multivariate Linear Regression was used to predict fare from other passenger features.

Results:

| Metric | Value |
|---|---:|
| MAE | 19.6457 |
| RMSE | 41.2628 |
| R² | 0.3474 |
| Adjusted R² | 0.3124 |

The residual analysis showed evidence of heteroscedasticity. The residual spread changes with predicted fare values rather than maintaining a constant variance.

## Final Model

Among the three required classifiers, Logistic Regression was selected as the final classifier based on its overall metric balance and highest AUC.

The complete fitted preprocessing and classification pipeline is saved as:

`final_pipeline.joblib`

The saved pipeline was reloaded using `joblib.load` and successfully generated a prediction from a raw passenger record.

## Module Outputs

The module generates:

- Raw Titanic dataset
- Cleaned Titanic dataset
- Missing-value analysis
- EDA report
- Standardization analysis
- Survival analysis outputs
- Classification metrics
- Confusion matrices
- ROC curves
- Decision tree visualization
- Class balance analysis
- Imbalance comparison
- Random Forest GridSearch results
- Regression metrics
- Regression residual plot
- Final model comparison
- Reloadable fitted classification pipeline

## Completion Status
Module 2 analytics and modeling pipeline completed successfully.
