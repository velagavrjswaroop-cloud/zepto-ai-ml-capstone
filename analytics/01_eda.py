import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.join(BASE_DIR, "charts")

os.makedirs(CHART_DIR, exist_ok=True)

df = sns.load_dataset("titanic")

df.to_csv(
    os.path.join(BASE_DIR, "titanic.csv"),
    index=False
)

print("Shape:", df.shape)

print("\nData Information:")
df.info()

print("\nDescriptive Statistics:")
print(df.describe(include="all"))

missing = df.isnull().sum()
missing_percent = df.isnull().mean() * 100

missing_table = pd.DataFrame({
    "missing_count": missing,
    "missing_percentage": missing_percent
})

missing_table = missing_table[
    missing_table["missing_count"] > 0
].sort_values(
    "missing_percentage",
    ascending=False
)

print("\nMissing Values:")
print(missing_table)

missing_table.to_csv(
    os.path.join(BASE_DIR, "missing_values.csv")
)

cleaned = df.copy()

high_missing_columns = [
    column
    for column in cleaned.columns
    if cleaned[column].isnull().mean() > 0.30
]

cleaned = cleaned.drop(
    columns=high_missing_columns
)

remaining_missing_columns = [
    column
    for column in cleaned.columns
    if cleaned[column].isnull().any()
]

for column in remaining_missing_columns:
    missing_percentage = cleaned[column].isnull().mean() * 100

    if missing_percentage < 5:
        cleaned = cleaned.dropna(subset=[column])

    elif missing_percentage <= 30:
        if pd.api.types.is_numeric_dtype(cleaned[column]):
            cleaned[column] = cleaned[column].fillna(
                cleaned[column].median()
            )
        else:
            cleaned[column] = cleaned[column].fillna(
                cleaned[column].mode()[0]
            )

    else:
        cleaned[column] = cleaned[column].fillna(
            cleaned[column].mode()[0]
        )

print("\nHigh Missing Columns Dropped:")
print(high_missing_columns)

print("\nCleaned Shape:", cleaned.shape)

print("\nRows Removed:", len(df) - len(cleaned))

print("\nRemaining Missing Values:")
print(cleaned.isnull().sum())

age_q1 = cleaned["age"].quantile(0.25)
age_q3 = cleaned["age"].quantile(0.75)
age_iqr = age_q3 - age_q1

age_lower = age_q1 - 1.5 * age_iqr
age_upper = age_q3 + 1.5 * age_iqr

age_outliers = (
    (cleaned["age"] < age_lower) |
    (cleaned["age"] > age_upper)
).sum()

fare_q1 = cleaned["fare"].quantile(0.25)
fare_q3 = cleaned["fare"].quantile(0.75)
fare_iqr = fare_q3 - fare_q1

fare_lower = fare_q1 - 1.5 * fare_iqr
fare_upper = fare_q3 + 1.5 * fare_iqr

fare_outliers = (
    (cleaned["fare"] < fare_lower) |
    (cleaned["fare"] > fare_upper)
).sum()

print("\nIQR Outlier Counts:")
print("Age:", age_outliers)
print("Fare:", fare_outliers)

fare_mean = cleaned["fare"].mean()
fare_median = cleaned["fare"].median()
fare_mode = cleaned["fare"].mode()[0]

if fare_mean > fare_median > fare_mode:
    skewness_conclusion = "Fare is right-skewed."
elif fare_mean < fare_median < fare_mode:
    skewness_conclusion = "Fare is left-skewed."
else:
    skewness_conclusion = "Fare does not show a clear skewness pattern from mean, median and mode ordering."

print("\nFare Statistics:")
print("Mean:", fare_mean)
print("Median:", fare_median)
print("Mode:", fare_mode)
print("Skewness Conclusion:", skewness_conclusion)

female_mask = cleaned["sex"] == "female"
male_mask = cleaned["sex"] == "male"

first_class_mask = cleaned["pclass"] == 1
second_class_mask = cleaned["pclass"] == 2
third_class_mask = cleaned["pclass"] == 3

female_first_mask = female_mask & first_class_mask
female_second_mask = female_mask & second_class_mask
female_third_mask = female_mask & third_class_mask

male_first_mask = male_mask & first_class_mask
male_second_mask = male_mask & second_class_mask
male_third_mask = male_mask & third_class_mask

survival_by_sex = pd.Series({
    "female": cleaned.loc[female_mask, "survived"].mean(),
    "male": cleaned.loc[male_mask, "survived"].mean()
})

survival_by_class = pd.Series({
    1: cleaned.loc[first_class_mask, "survived"].mean(),
    2: cleaned.loc[second_class_mask, "survived"].mean(),
    3: cleaned.loc[third_class_mask, "survived"].mean()
})

survival_by_sex_class = pd.Series({
    "female_1": cleaned.loc[female_first_mask, "survived"].mean(),
    "female_2": cleaned.loc[female_second_mask, "survived"].mean(),
    "female_3": cleaned.loc[female_third_mask, "survived"].mean(),
    "male_1": cleaned.loc[male_first_mask, "survived"].mean(),
    "male_2": cleaned.loc[male_second_mask, "survived"].mean(),
    "male_3": cleaned.loc[male_third_mask, "survived"].mean()
})

print("\nSurvival by Sex:")
print(survival_by_sex)

print("\nSurvival by Pclass:")
print(survival_by_class)

print("\nSurvival by Sex and Pclass:")
print(survival_by_sex_class)

plt.figure(figsize=(8, 6))
sns.histplot(cleaned["age"], kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(
    os.path.join(CHART_DIR, "age_histogram.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.close()

plt.figure(figsize=(8, 6))
sns.boxplot(x=cleaned["age"])
plt.title("Age Box Plot")
plt.xlabel("Age")
plt.tight_layout()
plt.savefig(
    os.path.join(CHART_DIR, "age_boxplot.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.close()

plt.figure(figsize=(8, 6))
sns.histplot(cleaned["fare"], kde=True)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(
    os.path.join(CHART_DIR, "fare_histogram.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.close()

plt.figure(figsize=(8, 6))
sns.boxplot(x=cleaned["fare"])
plt.title("Fare Box Plot")
plt.xlabel("Fare")
plt.tight_layout()
plt.savefig(
    os.path.join(CHART_DIR, "fare_boxplot.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.close()

survival_sex_class = cleaned.groupby(
    ["sex", "pclass"]
)["survived"].mean().reset_index()

plt.figure(figsize=(9, 6))
sns.barplot(
    data=survival_sex_class,
    x="pclass",
    y="survived",
    hue="sex"
)
plt.title("Survival Rate by Sex and Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(
    os.path.join(CHART_DIR, "survival_by_sex_class.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.close()

plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=cleaned,
    x="age",
    y="fare",
    hue="survived",
    style="survived",
    alpha=0.7
)
plt.title("Age, Fare and Survival")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.tight_layout()
plt.savefig(
    os.path.join(CHART_DIR, "age_fare_survival.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.close()

plt.figure(figsize=(9, 6))
sns.boxplot(
    data=cleaned,
    x="pclass",
    y="fare",
    hue="survived"
)
plt.title("Fare by Passenger Class and Survival")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")
plt.tight_layout()
plt.savefig(
    os.path.join(CHART_DIR, "fare_class_survival.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.close()

plt.figure(figsize=(9, 6))
sns.countplot(
    data=cleaned,
    x="pclass",
    hue="sex"
)
plt.title("Passenger Count by Class and Sex")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(
    os.path.join(CHART_DIR, "class_sex_count.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.close()

correlation_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

correlation_matrix = cleaned[
    correlation_columns
].corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(9, 7))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".3f",
    cmap="coolwarm",
    center=0
)
plt.title("Titanic Correlation Matrix")
plt.tight_layout()
plt.savefig(
    os.path.join(CHART_DIR, "correlation_heatmap.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.close()

correlation_pairs = []

for i in range(len(correlation_columns)):
    for j in range(i + 1, len(correlation_columns)):
        variable_1 = correlation_columns[i]
        variable_2 = correlation_columns[j]
        value = correlation_matrix.loc[
            variable_1,
            variable_2
        ]

        correlation_pairs.append({
            "variable_1": variable_1,
            "variable_2": variable_2,
            "correlation": value,
            "absolute_correlation": abs(value)
        })

correlation_pairs_df = pd.DataFrame(
    correlation_pairs
).sort_values(
    "absolute_correlation",
    ascending=False
)

top_two_correlations = correlation_pairs_df.head(2)

print("\nTop Two Off-Diagonal Correlations:")
print(top_two_correlations)

scaler = StandardScaler()

standardized_age_fare = scaler.fit_transform(
    cleaned[["age", "fare"]]
)

standardization_check = pd.DataFrame({
    "variable": ["age", "fare"],
    "before_mean": [
        cleaned["age"].mean(),
        cleaned["fare"].mean()
    ],
    "before_std": [
        cleaned["age"].std(),
        cleaned["fare"].std()
    ],
    "after_mean": [
        standardized_age_fare[:, 0].mean(),
        standardized_age_fare[:, 1].mean()
    ],
    "after_std": [
        standardized_age_fare[:, 0].std(),
        standardized_age_fare[:, 1].std()
    ]
})

print("\nStandardization Check:")
print(standardization_check)

standardization_check.to_csv(
    os.path.join(BASE_DIR, "standardization_check.csv"),
    index=False
)

cleaned.to_csv(
    os.path.join(BASE_DIR, "cleaned_titanic.csv"),
    index=False
)

female_first_rate = cleaned.loc[
    female_first_mask,
    "survived"
].mean()

female_second_rate = cleaned.loc[
    female_second_mask,
    "survived"
].mean()

female_third_rate = cleaned.loc[
    female_third_mask,
    "survived"
].mean()

male_first_rate = cleaned.loc[
    male_first_mask,
    "survived"
].mean()

male_second_rate = cleaned.loc[
    male_second_mask,
    "survived"
].mean()

male_third_rate = cleaned.loc[
    male_third_mask,
    "survived"
].mean()

with open(
    os.path.join(BASE_DIR, "eda_report.md"),
    "w",
    encoding="utf-8"
) as file:
    file.write(
        f"""# Titanic Exploratory Data Analysis Report

## Dataset

The Titanic dataset contains {df.shape[0]} rows and {df.shape[1]} columns before cleaning.

The raw dataset was loaded once using seaborn and saved as `titanic.csv` for offline use.

## Missing Values

The `deck` column had {missing_percent["deck"]:.2f}% missing values and was removed because the missing proportion was above 30%.

The `age` column had {missing_percent["age"]:.2f}% missing values. Because this was between 5% and 30%, missing values were handled using median imputation.

The `embarked` column had {missing_percent["embarked"]:.2f}% missing values. Because this was below 5%, rows with missing values were removed.

The `embark_town` column had {missing_percent["embark_town"]:.2f}% missing values. Because this was below 5%, rows with missing values were removed.

After cleaning, the dataset contains {len(cleaned)} rows and has no remaining missing values.

## IQR Outliers

Age outliers using the 1.5 IQR rule: {age_outliers}.

Fare outliers using the 1.5 IQR rule: {fare_outliers}.

## Fare Distribution

Fare mean: {fare_mean:.4f}.

Fare median: {fare_median:.4f}.

Fare mode: {fare_mode:.4f}.

{skewness_conclusion}

## Bivariate Survival Analysis

Female survival rate: {survival_by_sex["female"]:.4f}.

Male survival rate: {survival_by_sex["male"]:.4f}.

First-class survival rate: {survival_by_class[1]:.4f}.

Second-class survival rate: {survival_by_class[2]:.4f}.

Third-class survival rate: {survival_by_class[3]:.4f}.

Female first-class survival rate: {female_first_rate:.4f}.

Female second-class survival rate: {female_second_rate:.4f}.

Female third-class survival rate: {female_third_rate:.4f}.

Male first-class survival rate: {male_first_rate:.4f}.

Male second-class survival rate: {male_second_rate:.4f}.

Male third-class survival rate: {male_third_rate:.4f}.

The sex and passenger-class survival rates were calculated using boolean masks with `&` combinations.

## Correlation Analysis

The strongest off-diagonal correlation is between {top_two_correlations.iloc[0]["variable_1"]} and {top_two_correlations.iloc[0]["variable_2"]}, with correlation {top_two_correlations.iloc[0]["correlation"]:.4f}. This indicates a moderate negative relationship between passenger class and fare.

The second strongest off-diagonal correlation is between {top_two_correlations.iloc[1]["variable_1"]} and {top_two_correlations.iloc[1]["variable_2"]}, with correlation {top_two_correlations.iloc[1]["correlation"]:.4f}. This indicates a moderate positive relationship between siblings/spouses and parents/children aboard.

## Multivariate Chart Interpretations

### Survival by Sex and Passenger Class

Survival rates vary substantially by both sex and passenger class. Female passengers generally have higher survival rates than male passengers, while higher passenger classes also show better survival outcomes.

### Age, Fare and Survival

The scatter plot shows the joint relationship between age, fare and survival. Surviving passengers are more concentrated in higher-fare observations, while fare and age distributions contain substantial variation.

### Fare by Passenger Class and Survival

Fare distributions differ considerably across passenger classes, with higher classes generally associated with higher fares. Survival also varies within each class, demonstrating that class and survival are jointly informative.

### Passenger Count by Class and Sex

Passenger composition differs across classes and sex groups. Third class contains the largest number of passengers, while the sex distribution also varies across passenger classes.

## Standardization

Age and fare were standardized using StandardScaler on the full cleaned dataset for exploratory analysis only. Their standardized means are approximately zero and their standard deviations are approximately one.

This standardized data was not used for model training.

## Output Charts

The `charts` directory contains the histogram, box plot, multivariate visualizations, and correlation heatmap.
"""
    )

print("\nEDA completed successfully.")
print("Age outliers:", age_outliers)
print("Fare outliers:", fare_outliers)
print(
    "Strongest correlation:",
    top_two_correlations.iloc[0]["variable_1"],
    "vs",
    top_two_correlations.iloc[0]["variable_2"],
    "=",
    round(top_two_correlations.iloc[0]["correlation"], 4)
)
print(
    "Second strongest correlation:",
    top_two_correlations.iloc[1]["variable_1"],
    "vs",
    top_two_correlations.iloc[1]["variable_2"],
    "=",
    round(top_two_correlations.iloc[1]["correlation"], 4)
)