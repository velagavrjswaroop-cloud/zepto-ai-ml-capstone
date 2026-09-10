# Titanic Modeling Report

## Dataset and Split

The cleaned Titanic dataset was loaded from `cleaned_titanic.csv`.

Classification used a stratified 80/20 train-test split with random state 42. Stratification was used because the survival target contains two classes and the class proportions should remain similar in the training and test sets.

Training rows: 711

Test rows: 178

Training survival rate: 0.3826

Test survival rate: 0.3820

## Class Balance

 survived  count  percentage
        0    549   61.754781
        1    340   38.245219

## Classification Results

              model  accuracy  precision  recall     f1    auc
Logistic Regression    0.8090     0.7833  0.6912 0.7344 0.8610
      Decision Tree    0.7640     0.7600  0.5588 0.6441 0.8374
      Random Forest    0.8034     0.7619  0.7059 0.7328 0.8237

All three classifiers used the identical train-test split and preprocessing was fitted only on the training data through pipelines.

## Imbalance Experiment

             strategy  precision  recall     f1
             Baseline     0.7833  0.6912 0.7344
Class Weight Balanced     0.7183  0.7500 0.7338
                SMOTE     0.7353  0.7353 0.7353

The strongest F1 score in the imbalance experiment was obtained by SMOTE with an F1 score of 0.7353. Precision was 0.7353 and recall was 0.7353.

## Random Forest Grid Search

Best parameters:

{'model__max_depth': 5, 'model__max_features': 'sqrt', 'model__n_estimators': 200}

Best cross-validation F1 score: 0.7408

Out-of-bag score: 0.8214

Tuned Random Forest test accuracy: 0.8315

Tuned Random Forest test precision: 0.8654

Tuned Random Forest test recall: 0.6618

Tuned Random Forest test F1: 0.7500

Tuned Random Forest test AUC: 0.8389

## Regression Results

                         model     mae    rmse     r2  adjusted_r2
Multivariate Linear Regression 19.6457 41.2628 0.3474       0.3124

The residual plot was saved as `charts/regression_residual_plot.png`.

Spearman correlation between absolute residuals and predicted fare: 0.4389

Spearman p-value: 0.0000

The residual spread shows evidence of heteroscedasticity.

## Final Recommendation

Logistic Regression is the recommended classifier because it achieved the highest F1 score of 0.734 among the three required classifiers. Its accuracy was 0.809, precision was 0.783, recall was 0.691, and ROC/AUC was 0.861. The imbalance experiment showed that SMOTE produced the strongest F1 score of 0.735 among the three imbalance strategies tested. The selected pipeline includes preprocessing and the final estimator together, so it can accept raw passenger records after reloading.

## Saved Pipeline

The complete fitted classification pipeline was saved as `final_pipeline.joblib`.

The saved pipeline was reloaded successfully and tested using a raw passenger record.

New passenger prediction: 0

Predicted survival probability: 0.0947
