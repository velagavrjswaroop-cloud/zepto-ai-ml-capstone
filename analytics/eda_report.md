# Titanic Exploratory Data Analysis Report

## Dataset

The Titanic dataset contains 891 rows and 15 columns before cleaning.

The raw dataset was loaded once using seaborn and saved as `titanic.csv` for offline use.

## Missing Values

The `deck` column had 77.22% missing values and was removed because the missing proportion was above 30%.

The `age` column had 19.87% missing values. Because this was between 5% and 30%, missing values were handled using median imputation.

The `embarked` column had 0.22% missing values. Because this was below 5%, rows with missing values were removed.

The `embark_town` column had 0.22% missing values. Because this was below 5%, rows with missing values were removed.

After cleaning, the dataset contains 889 rows and has no remaining missing values.

## IQR Outliers

Age outliers using the 1.5 IQR rule: 65.

Fare outliers using the 1.5 IQR rule: 114.

## Fare Distribution

Fare mean: 32.0967.

Fare median: 14.4542.

Fare mode: 8.0500.

Fare is right-skewed.

## Bivariate Survival Analysis

Female survival rate: 0.7404.

Male survival rate: 0.1889.

First-class survival rate: 0.6262.

Second-class survival rate: 0.4728.

Third-class survival rate: 0.2424.

Female first-class survival rate: 0.9674.

Female second-class survival rate: 0.9211.

Female third-class survival rate: 0.5000.

Male first-class survival rate: 0.3689.

Male second-class survival rate: 0.1574.

Male third-class survival rate: 0.1354.

The sex and passenger-class survival rates were calculated using boolean masks with `&` combinations.

## Correlation Analysis

The strongest off-diagonal correlation is between pclass and fare, with correlation -0.5482. This indicates a moderate negative relationship between passenger class and fare.

The second strongest off-diagonal correlation is between sibsp and parch, with correlation 0.4145. This indicates a moderate positive relationship between siblings/spouses and parents/children aboard.

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
