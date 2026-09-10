import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.base import clone
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from scipy.stats import spearmanr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.join(BASE_DIR, "charts")
os.makedirs(CHART_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(BASE_DIR, "cleaned_titanic.csv"))

target = "survived"
features = ["pclass", "age", "sibsp", "parch", "fare", "sex", "embarked"]

X = df[features].copy()
y = df[target].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

numeric_features = ["pclass", "age", "sibsp", "parch", "fare"]
categorical_features = ["sex", "embarked"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
}

fitted_models = {}
classification_results = []

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for index, (name, estimator) in enumerate(models.items()):
    pipeline = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("model", estimator)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    auc = roc_auc_score(y_test, probabilities)

    classification_results.append({
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    })

    fitted_models[name] = pipeline

    cm = confusion_matrix(y_test, predictions)
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not Survived", "Survived"]
    ).plot(ax=axes[index], values_format="d")
    axes[index].set_title(name)

plt.tight_layout()
plt.savefig(
    os.path.join(CHART_DIR, "classification_confusion_matrices.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.close()

classification_results_df = pd.DataFrame(classification_results)
classification_results_df.to_csv(
    os.path.join(BASE_DIR, "model_comparison.csv"),
    index=False
)

plt.figure(figsize=(9, 6))

for name, pipeline in fitted_models.items():
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = __import__("sklearn").metrics.roc_curve(y_test, probabilities)
    auc = roc_auc_score(y_test, probabilities)
    plt.plot(fpr, tpr, label=f"{name} AUC={auc:.3f}")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(CHART_DIR, "classification_roc_curves.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.close()

best_model_name = classification_results_df.sort_values(
    ["f1", "auc"],
    ascending=False
).iloc[0]["model"]

best_pipeline = fitted_models[best_model_name]

tree_pipeline = fitted_models["Decision Tree"]
tree_model = tree_pipeline.named_steps["model"]
tree_preprocessor = tree_pipeline.named_steps["preprocessor"]

feature_names = tree_preprocessor.get_feature_names_out()

plt.figure(figsize=(24, 14))
plot_tree(
    tree_model,
    feature_names=feature_names,
    class_names=["Not Survived", "Survived"],
    filled=True,
    rounded=True,
    max_depth=4,
    fontsize=8
)
plt.title("Decision Tree")
plt.tight_layout()
plt.savefig(
    os.path.join(CHART_DIR, "decision_tree.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.close()

class_balance = y.value_counts().rename_axis("survived").reset_index(name="count")
class_balance["percentage"] = (
    class_balance["count"] / len(y) * 100
)
class_balance.to_csv(
    os.path.join(BASE_DIR, "class_balance.csv"),
    index=False
)

imbalance_models = {
    "Baseline": LogisticRegression(max_iter=1000, random_state=42),
    "Class Weight Balanced": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
}

imbalance_results = []

for name, estimator in imbalance_models.items():
    pipeline = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("model", estimator)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    imbalance_results.append({
        "strategy": name,
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0)
    })

smote_pipeline = ImbPipeline([
    ("preprocessor", clone(preprocessor)),
    ("smote", SMOTE(random_state=42)),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])

smote_pipeline.fit(X_train, y_train)

smote_predictions = smote_pipeline.predict(X_test)

imbalance_results.append({
    "strategy": "SMOTE",
    "precision": precision_score(y_test, smote_predictions, zero_division=0),
    "recall": recall_score(y_test, smote_predictions, zero_division=0),
    "f1": f1_score(y_test, smote_predictions, zero_division=0)
})

imbalance_results_df = pd.DataFrame(imbalance_results)

imbalance_results_df.to_csv(
    os.path.join(BASE_DIR, "imbalance_comparison.csv"),
    index=False
)

grid_pipeline = Pipeline([
    ("preprocessor", clone(preprocessor)),
    ("model", RandomForestClassifier(
        oob_score=True,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    ))
])

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10],
    "model__max_features": ["sqrt", "log2"]
}

grid_search = GridSearchCV(
    estimator=grid_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    refit=True
)

grid_search.fit(X_train, y_train)

best_rf_pipeline = grid_search.best_estimator_
best_rf_model = best_rf_pipeline.named_steps["model"]

grid_best_params = grid_search.best_params_
grid_best_score = grid_search.best_score_
oob_score = best_rf_model.oob_score_

tuned_rf_predictions = best_rf_pipeline.predict(X_test)
tuned_rf_probabilities = best_rf_pipeline.predict_proba(X_test)[:, 1]

tuned_rf_metrics = {
    "accuracy": accuracy_score(y_test, tuned_rf_predictions),
    "precision": precision_score(y_test, tuned_rf_predictions, zero_division=0),
    "recall": recall_score(y_test, tuned_rf_predictions, zero_division=0),
    "f1": f1_score(y_test, tuned_rf_predictions, zero_division=0),
    "auc": roc_auc_score(y_test, tuned_rf_probabilities)
}

regression_features = ["pclass", "age", "sibsp", "parch", "sex", "embarked"]
X_reg = df[regression_features].copy()
y_reg = df["fare"].copy()

X_reg_train = X_reg.loc[X_train.index]
X_reg_test = X_reg.loc[X_test.index]
y_reg_train = y_reg.loc[X_train.index]
y_reg_test = y_reg.loc[X_test.index]

reg_numeric_features = ["pclass", "age", "sibsp", "parch"]
reg_categorical_features = ["sex", "embarked"]

reg_numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

reg_categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

reg_preprocessor = ColumnTransformer([
    ("numeric", reg_numeric_pipeline, reg_numeric_features),
    ("categorical", reg_categorical_pipeline, reg_categorical_features)
])

regression_pipeline = Pipeline([
    ("preprocessor", reg_preprocessor),
    ("model", LinearRegression())
])

regression_pipeline.fit(X_reg_train, y_reg_train)

reg_predictions = regression_pipeline.predict(X_reg_test)

mae = mean_absolute_error(y_reg_test, reg_predictions)
rmse = np.sqrt(mean_squared_error(y_reg_test, reg_predictions))
r2 = r2_score(y_reg_test, reg_predictions)

reg_transformed_train = regression_pipeline.named_steps[
    "preprocessor"
].transform(X_reg_train)

p = reg_transformed_train.shape[1]
n = len(y_reg_test)

adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

residuals = y_reg_test - reg_predictions

spearman_correlation, spearman_pvalue = spearmanr(
    np.abs(residuals),
    reg_predictions
)

if spearman_pvalue < 0.05 and spearman_correlation > 0.20:
    heteroscedasticity_conclusion = "The residual spread shows evidence of heteroscedasticity."
else:
    heteroscedasticity_conclusion = "The residual spread does not show strong statistical evidence of heteroscedasticity."

plt.figure(figsize=(9, 6))
plt.scatter(reg_predictions, residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Fare")
plt.ylabel("Residual")
plt.title("Regression Residual Plot")
plt.tight_layout()
plt.savefig(
    os.path.join(CHART_DIR, "regression_residual_plot.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.close()

regression_results = pd.DataFrame([{
    "model": "Multivariate Linear Regression",
    "mae": mae,
    "rmse": rmse,
    "r2": r2,
    "adjusted_r2": adjusted_r2
}])

regression_results.to_csv(
    os.path.join(BASE_DIR, "regression_results.csv"),
    index=False
)

final_comparison = classification_results_df.copy()
final_comparison["mae"] = np.nan
final_comparison["rmse"] = np.nan
final_comparison["r2"] = np.nan
final_comparison["adjusted_r2"] = np.nan

regression_row = pd.DataFrame([{
    "model": "Multivariate Linear Regression",
    "accuracy": np.nan,
    "precision": np.nan,
    "recall": np.nan,
    "f1": np.nan,
    "auc": np.nan,
    "mae": mae,
    "rmse": rmse,
    "r2": r2,
    "adjusted_r2": adjusted_r2
}])

final_comparison = pd.concat(
    [final_comparison, regression_row],
    ignore_index=True
)

final_comparison.to_csv(
    os.path.join(BASE_DIR, "final_model_comparison.csv"),
    index=False
)

pipeline_path = os.path.join(BASE_DIR, "final_pipeline.joblib")
joblib.dump(best_pipeline, pipeline_path)

loaded_pipeline = joblib.load(pipeline_path)

new_passenger = pd.DataFrame([{
    "pclass": 3,
    "age": 22,
    "sibsp": 1,
    "parch": 0,
    "fare": 7.25,
    "sex": "male",
    "embarked": "S"
}])

new_prediction = loaded_pipeline.predict(new_passenger)[0]
new_probability = loaded_pipeline.predict_proba(new_passenger)[0, 1]

imbalance_best = imbalance_results_df.sort_values(
    "f1",
    ascending=False
).iloc[0]

classification_recommendation = (
    f"{best_model_name} is the recommended classifier because it achieved the highest F1 score "
    f"of {classification_results_df.loc[classification_results_df['model'] == best_model_name, 'f1'].iloc[0]:.3f} "
    f"among the three required classifiers. Its accuracy was "
    f"{classification_results_df.loc[classification_results_df['model'] == best_model_name, 'accuracy'].iloc[0]:.3f}, "
    f"precision was "
    f"{classification_results_df.loc[classification_results_df['model'] == best_model_name, 'precision'].iloc[0]:.3f}, "
    f"recall was "
    f"{classification_results_df.loc[classification_results_df['model'] == best_model_name, 'recall'].iloc[0]:.3f}, "
    f"and ROC/AUC was "
    f"{classification_results_df.loc[classification_results_df['model'] == best_model_name, 'auc'].iloc[0]:.3f}. "
    f"The imbalance experiment showed that {imbalance_best['strategy']} produced the strongest F1 score "
    f"of {imbalance_best['f1']:.3f} among the three imbalance strategies tested. "
    f"The selected pipeline includes preprocessing and the final estimator together, so it can accept raw passenger records after reloading."
)

report = f"""# Titanic Modeling Report

## Dataset and Split

The cleaned Titanic dataset was loaded from `cleaned_titanic.csv`.

Classification used a stratified 80/20 train-test split with random state 42. Stratification was used because the survival target contains two classes and the class proportions should remain similar in the training and test sets.

Training rows: {len(X_train)}

Test rows: {len(X_test)}

Training survival rate: {y_train.mean():.4f}

Test survival rate: {y_test.mean():.4f}

## Class Balance

{class_balance.to_string(index=False)}

## Classification Results

{classification_results_df.round(4).to_string(index=False)}

All three classifiers used the identical train-test split and preprocessing was fitted only on the training data through pipelines.

## Imbalance Experiment

{imbalance_results_df.round(4).to_string(index=False)}

The strongest F1 score in the imbalance experiment was obtained by {imbalance_best['strategy']} with an F1 score of {imbalance_best['f1']:.4f}. Precision was {imbalance_best['precision']:.4f} and recall was {imbalance_best['recall']:.4f}.

## Random Forest Grid Search

Best parameters:

{grid_best_params}

Best cross-validation F1 score: {grid_best_score:.4f}

Out-of-bag score: {oob_score:.4f}

Tuned Random Forest test accuracy: {tuned_rf_metrics['accuracy']:.4f}

Tuned Random Forest test precision: {tuned_rf_metrics['precision']:.4f}

Tuned Random Forest test recall: {tuned_rf_metrics['recall']:.4f}

Tuned Random Forest test F1: {tuned_rf_metrics['f1']:.4f}

Tuned Random Forest test AUC: {tuned_rf_metrics['auc']:.4f}

## Regression Results

{regression_results.round(4).to_string(index=False)}

The residual plot was saved as `charts/regression_residual_plot.png`.

Spearman correlation between absolute residuals and predicted fare: {spearman_correlation:.4f}

Spearman p-value: {spearman_pvalue:.4f}

{heteroscedasticity_conclusion}

## Final Recommendation

{classification_recommendation}

## Saved Pipeline

The complete fitted classification pipeline was saved as `final_pipeline.joblib`.

The saved pipeline was reloaded successfully and tested using a raw passenger record.

New passenger prediction: {int(new_prediction)}

Predicted survival probability: {new_probability:.4f}
"""

with open(
    os.path.join(BASE_DIR, "modeling_report.md"),
    "w",
    encoding="utf-8"
) as file:
    file.write(report)

print("\nClassification Results")
print(classification_results_df.round(4).to_string(index=False))

print("\nClass Balance")
print(class_balance.to_string(index=False))

print("\nImbalance Results")
print(imbalance_results_df.round(4).to_string(index=False))

print("\nGrid Search Best Parameters")
print(grid_best_params)

print(f"\nGrid Search Best F1: {grid_best_score:.4f}")
print(f"OOB Score: {oob_score:.4f}")

print("\nTuned Random Forest Test Metrics")
for key, value in tuned_rf_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nRegression Results")
print(regression_results.round(4).to_string(index=False))

print(f"\nSpearman correlation: {spearman_correlation:.4f}")
print(f"Spearman p-value: {spearman_pvalue:.4f}")
print(heteroscedasticity_conclusion)

print(f"\nRecommended classifier: {best_model_name}")
print(f"New passenger prediction after pipeline reload: {int(new_prediction)}")
print(f"New passenger survival probability: {new_probability:.4f}")
print(f"\nPipeline saved to: {pipeline_path}")