# ============================================================
# AAPL STOCK PRICE ANALYSIS & FORECASTING PROJECT
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# 1. PROBLEM DEFINITION
# ============================================================

print("=" * 80)
print("APPLE STOCK PRICE ANALYSIS & FORECASTING PROJECT")
print("=" * 80)

print("""
OBJECTIVE

Analyze the AAPL stock dataset containing
approximately 60 technical indicators.

The goal is to:

1. Explore the dataset
2. Understand feature relationships
3. Perform data preparation
4. Select important features
5. Transform the data
6. Build machine learning models
7. Forecast future stock prices
""")

# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading Dataset...")

try:
    df = pd.read_csv("AAPL.csv")
except Exception as e:
    print("ERROR LOADING DATASET")
    print(e)
    exit()

print("Dataset Loaded Successfully")
print("Dataset Shape:", df.shape)

# ============================================================
# 3. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 80)

print("\nFirst 5 Records")
print(df.head())

print("\nDataset Information")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

# ------------------------------------------------------------
# Missing Values
# ------------------------------------------------------------

print("\nMissing Values")

missing = df.isnull().sum()

missing_df = pd.DataFrame({
    "Column": missing.index,
    "Missing Values": missing.values
})

print(
    missing_df[
        missing_df["Missing Values"] > 0
    ]
)

# ------------------------------------------------------------
# Duplicate Records
# ------------------------------------------------------------

duplicates = df.duplicated().sum()

print("\nDuplicate Rows:", duplicates)

# ------------------------------------------------------------
# Correlation Heatmap
# ------------------------------------------------------------

print("\nGenerating Correlation Heatmap...")

numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(14,10))

sns.heatmap(
    numeric_df.corr(),
    cmap="coolwarm",
    center=0
)

plt.title("Feature Correlation Matrix")

plt.tight_layout()

plt.savefig(
    "correlation_heatmap.png",
    dpi=300
)

plt.close()

print("Saved: correlation_heatmap.png")

# ------------------------------------------------------------
# Feature Distribution
# ------------------------------------------------------------

print("Generating Feature Distribution Plot...")

numeric_df.hist(
    figsize=(16,12),
    bins=20
)

plt.tight_layout()

plt.savefig(
    "feature_distribution.png",
    dpi=300
)

plt.close()

print("Saved: feature_distribution.png")

# ============================================================
# 4. DATA PREPARATION
# ============================================================

print("\n" + "=" * 80)
print("DATA PREPARATION")
print("=" * 80)

# Date Conversion

if "Date" in df.columns:

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df.sort_values(
        by="Date",
        inplace=True
    )

# Remove Duplicates

df.drop_duplicates(inplace=True)

# Missing Value Treatment

df = df.ffill().bfill()

print("Remaining Missing Values:",
      df.isnull().sum().sum())

print("Data Preparation Completed")

# ============================================================
# 5. FEATURE SELECTION
# ============================================================

print("\n" + "=" * 80)
print("FEATURE SELECTION")
print("=" * 80)

# Automatically detect target

possible_targets = [
    "Close_forcast",
    "Close Forecast",
    "Close",
    "Adj Close"
]

TARGET = None

for col in possible_targets:
    if col in df.columns:
        TARGET = col
        break

if TARGET is None:
    print("\nUnable to locate target column.")
    print(df.columns.tolist())
    exit()

print("Target Column:", TARGET)

numeric_df = df.select_dtypes(include=np.number)

corr_matrix = numeric_df.corr()

target_corr = (
    corr_matrix[TARGET]
    .abs()
    .sort_values(
        ascending=False
    )
)

print("\nTop 20 Correlated Features")

print(
    target_corr.head(20)
)

selected_features = target_corr[
    target_corr > 0.30
].index.tolist()

print("\nSelected Features")

for feature in selected_features:
    print(feature)

# ============================================================
# 6. DATA TRANSFORMATION
# ============================================================

print("\n" + "=" * 80)
print("DATA TRANSFORMATION")
print("=" * 80)

drop_columns = [TARGET]

if "Date" in df.columns:
    drop_columns.append("Date")

X = df.drop(
    columns=drop_columns,
    errors="ignore"
)

X = X.select_dtypes(include=np.number)

y = df[TARGET]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(
    X_scaled,
    columns=X.columns
)

print("Feature Scaling Completed")

# ============================================================
# 7. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42
)

# ============================================================
# 8. MODEL BUILDING
# ============================================================

print("\n" + "=" * 80)
print("MODEL BUILDING")
print("=" * 80)

models = {

    "Linear Regression":
        LinearRegression(),

    "Ridge Regression":
        Ridge(alpha=1.0),

    "Lasso Regression":
        Lasso(alpha=0.001),

    "Elastic Net":
        ElasticNet(
            alpha=0.001,
            l1_ratio=0.5
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
}

# Optional XGBoost

try:

    from xgboost import XGBRegressor

    models["XGBoost"] = XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        objective="reg:squarederror",
        random_state=42
    )

    print("XGBoost Enabled")

except:

    print("XGBoost Not Installed")

# ============================================================
# 9. MODEL EVALUATION
# ============================================================

results = []

best_model = None
best_r2 = -999

for name, model in models.items():

    print(f"\nTraining {name}")

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append(
        [name, mae, rmse, r2]
    )

    print(
        f"MAE={mae:.4f} | "
        f"RMSE={rmse:.4f} | "
        f"R2={r2:.4f}"
    )

    if r2 > best_r2:
        best_r2 = r2
        best_model = model

# ============================================================
# MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "MAE",
        "RMSE",
        "R2"
    ]
)

results_df.sort_values(
    by="R2",
    ascending=False,
    inplace=True
)

print("\nMODEL COMPARISON")
print(results_df)

results_df.to_csv(
    "model_comparison.csv",
    index=False
)

# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 80)
print("FEATURE IMPORTANCE")
print("=" * 80)

if hasattr(best_model,
           "feature_importances_"):

    importance_df = pd.DataFrame({

        "Feature":
            X.columns,

        "Importance":
            best_model.feature_importances_
    })

    importance_df.sort_values(
        by="Importance",
        ascending=False,
        inplace=True
    )

    print(
        importance_df.head(20)
    )

    plt.figure(figsize=(10,6))

    sns.barplot(
        data=importance_df.head(15),
        x="Importance",
        y="Feature"
    )

    plt.title(
        "Top 15 Important Features"
    )

    plt.tight_layout()

    plt.savefig(
        "feature_importance.png",
        dpi=300
    )

    plt.close()

    print("Saved: feature_importance.png")

# ============================================================
# 11. FORECASTING
# ============================================================

print("\n" + "=" * 80)
print("1 WEEK FORECAST")
print("=" * 80)

future_data = X_scaled.tail(5)

forecast = best_model.predict(
    future_data
)

forecast_df = pd.DataFrame({

    "Forecast Day":
        [1,2,3,4,5],

    "Predicted Price":
        forecast
})

print(forecast_df)

forecast_df.to_csv(
    "forecast.csv",
    index=False
)

# ============================================================
# 12. PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("PROJECT COMPLETED")
print("=" * 80)

print("""
OUTPUT FILES GENERATED

1. correlation_heatmap.png
2. feature_distribution.png
3. feature_importance.png
4. model_comparison.csv
5. forecast.csv

PROJECT STEPS COMPLETED

✓ Problem Definition
✓ Exploratory Data Analysis
✓ Data Preparation
✓ Feature Selection
✓ Data Transformation
✓ Model Building
✓ Model Evaluation
✓ Feature Importance
✓ Forecasting
""")