from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline

from sklearn.model_selection import KFold, cross_validate

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

ORDERS_PATH = PROJECT_ROOT / "data" / "raw" / "orders.csv"
DELIVERIES_PATH = PROJECT_ROOT / "data" / "raw" / "deliveries.csv"
RESTAURANTS_PATH = PROJECT_ROOT / "data" / "raw" / "restaurants.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Load data
orders = pd.read_csv(ORDERS_PATH)
deliveries = pd.read_csv(DELIVERIES_PATH)
restaurants = pd.read_csv(RESTAURANTS_PATH)


# Filter to successfully delivered orders
delivered = deliveries[
    deliveries["delivery_status"] == "Delivered"
].copy()


print("Original delivery records:", len(deliveries))
print("Delivered records:", len(delivered))
print("Removed failed deliveries:", len(deliveries) - len(delivered))

print("\nDelivery time after filtering")
print("--------------------------------")
print(delivered["delivery_time_min"].describe())

print("\nZero delivery times after filtering")
print("--------------------------------")
print((delivered["delivery_time_min"] == 0).sum())

# Merge delivery data with order information
model_data = delivered.merge(
    orders,
    on="order_id",
    how="left"
)

# Merge restaurant information
model_data = model_data.merge(
    restaurants,
    on="restaurant_id",
    how="left"
)

print("\nModeling Dataset")
print("----------------")
print("Shape:", model_data.shape)

print("\nColumns")
print("-------")
print(model_data.columns.tolist())

print("\nMissing Values")
print("--------------")
print(model_data.isnull().sum())

print("\nDuplicate Rows")
print("--------------")
print(model_data.duplicated().sum())

# Convert order timestamp to datetime
model_data["order_timestamp"] = pd.to_datetime(
    model_data["order_timestamp"]
)

# Create time-based features
model_data["hour"] = model_data["order_timestamp"].dt.hour
model_data["day_of_week"] = model_data["order_timestamp"].dt.dayofweek
model_data["is_weekend"] = (
    model_data["day_of_week"] >= 5
).astype(int)


# Select modeling features
feature_columns = [
    "delivery_distance_km",
    "preparation_time_min",
    "order_amount",
    "num_items",
    "weather",
    "traffic_condition",
    "restaurant_category",
    "restaurant_rating",
    "avg_prep_time",
    "hour",
    "day_of_week",
    "is_weekend"
]

target_column = "delivery_time_min"

X = model_data[feature_columns].copy()
y = model_data[target_column].copy()


print("\nSelected Features")
print("-----------------")
print(X.columns.tolist())

print("\nTarget")
print("------")
print(target_column)

print("\nFeature Data Types")
print("------------------")
print(X.dtypes)

print("\nTarget Shape:", y.shape)
print("Feature Shape:", X.shape)

print("\nCategorical Features")
print("--------------------")
print(X.select_dtypes(include="object").columns.tolist())

print("\nNumerical Features")
print("------------------")
print(X.select_dtypes(exclude="object").columns.tolist())

# Target distribution
plt.figure(figsize=(8, 5))
plt.hist(y, bins=30)
plt.xlabel("Delivery Time (minutes)")
plt.ylabel("Number of Deliveries")
plt.title("Distribution of Delivery Time")
plt.tight_layout()

target_distribution_path = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "delivery_time_distribution.png"
)

plt.savefig(target_distribution_path)
plt.show()

print("\nTarget Skewness")
print("----------------")
print(y.skew())

numerical_features_for_eda = [
    "delivery_distance_km",
    "preparation_time_min",
    "avg_prep_time"
]

for feature in numerical_features_for_eda:
    print(f"\nCorrelation: {feature}")
    print(model_data[[feature, target_column]].corr().iloc[0, 1])

print("\nTarget Percentiles")
print("------------------")
print(y.quantile([0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.00]))

# Delivery time by traffic condition
print("\nDelivery Time by Traffic Condition")
print("-----------------------------------")

print(
    model_data.groupby("traffic_condition")["delivery_time_min"]
    .agg(["count", "mean", "median", "std"])
    .round(2)
)


# Delivery time by weather
print("\nDelivery Time by Weather")
print("------------------------")

print(
    model_data.groupby("weather")["delivery_time_min"]
    .agg(["count", "mean", "median", "std"])
    .round(2)
)


# Delivery time by restaurant category
print("\nDelivery Time by Restaurant Category")
print("-------------------------------------")

print(
    model_data.groupby("restaurant_category")["delivery_time_min"]
    .agg(["count", "mean", "median", "std"])
    .sort_values("mean", ascending=False)
    .round(2)
)

# ------------------------------
# Train/Test Split
# ------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTrain/Test Split")
print("----------------")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ------------------------------
# Preprocessing
# ------------------------------

numerical_features = [
    "delivery_distance_km",
    "preparation_time_min",
    "order_amount",
    "num_items",
    "restaurant_rating",
    "avg_prep_time",
    "hour",
    "day_of_week",
    "is_weekend"
]

categorical_features = [
    "weather",
    "traffic_condition",
    "restaurant_category"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# ------------------------------
# Baseline Linear Regression
# ------------------------------

baseline_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

baseline_model.fit(X_train, y_train)


# ------------------------------
# Predictions
# ------------------------------

y_pred = baseline_model.predict(X_test)


# ------------------------------
# Evaluation
# ------------------------------

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nBaseline Linear Regression")
print("-------------------------")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# ------------------------------
# Random Forest Regressor
# ------------------------------

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

random_forest_model.fit(X_train, y_train)


# ------------------------------
# Predictions
# ------------------------------

rf_pred = random_forest_model.predict(X_test)


# ------------------------------
# Evaluation
# ------------------------------

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("\nRandom Forest Regression")
print("-----------------------")
print(f"MAE  : {rf_mae:.4f}")
print(f"RMSE : {rf_rmse:.4f}")
print(f"R²   : {rf_r2:.4f}")

# ------------------------------
# Ridge Regression
# ------------------------------

ridge_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            Ridge(alpha=1.0)
        )
    ]
)

ridge_model.fit(X_train, y_train)


# ------------------------------
# Predictions
# ------------------------------

ridge_pred = ridge_model.predict(X_test)


# ------------------------------
# Evaluation
# ------------------------------

ridge_mae = mean_absolute_error(y_test, ridge_pred)
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
ridge_r2 = r2_score(y_test, ridge_pred)

print("\nRidge Regression")
print("----------------")
print(f"MAE  : {ridge_mae:.4f}")
print(f"RMSE : {ridge_rmse:.4f}")
print(f"R²   : {ridge_r2:.4f}")

# Lasso Regression
lasso_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", Lasso(alpha=1.0))
])

lasso_model.fit(X_train, y_train)

lasso_predictions = lasso_model.predict(X_test)

lasso_mae = mean_absolute_error(y_test, lasso_predictions)
lasso_rmse = mean_squared_error(y_test, lasso_predictions) ** 0.5
lasso_r2 = r2_score(y_test, lasso_predictions)

print("\nLasso Regression")
print("----------------")
print(f"MAE  : {lasso_mae:.4f}")
print(f"RMSE : {lasso_rmse:.4f}")
print(f"R²   : {lasso_r2:.4f}")

# 5-Fold Cross-Validation

cv = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Linear Regression": baseline_model,
    "Ridge Regression": ridge_model,
    "Random Forest": random_forest_model,
    "Lasso Regression": lasso_model
}

print("\n5-Fold Cross-Validation")
print("----------------------")

for name, model in models.items():

    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={
            "MAE": "neg_mean_absolute_error",
            "RMSE": "neg_root_mean_squared_error",
            "R2": "r2"
        },
        n_jobs=-1
    )

    mae = -cv_results["test_MAE"]
    rmse = -cv_results["test_RMSE"]
    r2 = cv_results["test_R2"]

    print(f"\n{name}")
    print(f"MAE  : {mae.mean():.4f} (+/- {mae.std():.4f})")
    print(f"RMSE : {rmse.mean():.4f} (+/- {rmse.std():.4f})")
    print(f"R²   : {r2.mean():.4f} (+/- {r2.std():.4f})")

# Feature Importance - Linear Regression Coefficients

linear_preprocessor = baseline_model.named_steps["preprocessor"]
linear_estimator = baseline_model.named_steps["regressor"]

feature_names = linear_preprocessor.get_feature_names_out()

coefficients = linear_estimator.coef_

feature_importance = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients
})

feature_importance["absolute_coefficient"] = (
    feature_importance["coefficient"].abs()
)

feature_importance = feature_importance.sort_values(
    "absolute_coefficient",
    ascending=False
)

print("\nTop 15 Features by Importance")
print("-----------------------------")
print(
    feature_importance[
        ["feature", "coefficient"]
    ].head(15).to_string(index=False)
)

# Actual vs Predicted Plot
y_pred = baseline_model.predict(X_test)

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.4)

# Perfect prediction line
min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Delivery Time (minutes)")
plt.ylabel("Predicted Delivery Time (minutes)")
plt.title("Actual vs Predicted Delivery Time")

plt.tight_layout()

plt.savefig(
    "outputs/figures/actual_vs_predicted_delivery_time.png",
    dpi=300
)

plt.show()

# Final model predictions
final_predictions = baseline_model.predict(X_test)

predictions_df = X_test.copy()
predictions_df["actual_delivery_time"] = y_test.values
predictions_df["predicted_delivery_time"] = final_predictions

print("\nFinal Prediction Summary")
print("------------------------")
print(predictions_df[
    ["actual_delivery_time", "predicted_delivery_time"]
].head())

predictions_path = OUTPUTS_DIR / "predictions" / "delivery_time_predictions.csv"

predictions_df.to_csv(predictions_path, index=False)

print(f"\nPredictions saved to: {predictions_path}")