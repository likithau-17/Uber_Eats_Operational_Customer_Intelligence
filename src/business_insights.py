from pathlib import Path

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "delivery_time_predictions.csv"
)


# Load delivery-time predictions
predictions = pd.read_csv(PREDICTIONS_PATH)


print("Delivery Prediction Business Analysis")
print("--------------------------------------")
print("Dataset shape:", predictions.shape)

print("\nColumns")
print("-------")
print(predictions.columns.tolist())

print("\nPreview")
print("-------")
print(predictions.head())

print("\nMissing Values")
print("--------------")
print(predictions.isnull().sum())

# ------------------------------
# Prediction Performance
# ------------------------------

predictions["prediction_error"] = (
    predictions["predicted_delivery_time"]
    - predictions["actual_delivery_time"]
)

predictions["absolute_error"] = (
    predictions["prediction_error"].abs()
)

mean_absolute_error = predictions["absolute_error"].mean()
mean_prediction_error = predictions["prediction_error"].mean()

print("\nPrediction Performance")
print("----------------------")
print(f"Mean Absolute Error: {mean_absolute_error:.2f} minutes")
print(f"Mean Prediction Error: {mean_prediction_error:.2f} minutes")

# ------------------------------
# Delivery Time by Traffic
# ------------------------------

print("\nDelivery Time by Traffic Condition")
print("-----------------------------------")

traffic_insights = (
    predictions.groupby("traffic_condition")[
        "actual_delivery_time"
    ]
    .agg(["count", "mean", "median"])
    .sort_values("mean", ascending=False)
    .round(2)
)

print(traffic_insights)


# ------------------------------
# Delivery Time by Weather
# ------------------------------

print("\nDelivery Time by Weather")
print("------------------------")

weather_insights = (
    predictions.groupby("weather")[
        "actual_delivery_time"
    ]
    .agg(["count", "mean", "median"])
    .sort_values("mean", ascending=False)
    .round(2)
)

print(weather_insights)

# ------------------------------
# Delivery Time by Distance
# ------------------------------

print("\nDelivery Time by Distance")
print("-------------------------")

predictions["distance_band"] = pd.cut(
    predictions["delivery_distance_km"],
    bins=[0, 2, 5, 10, float("inf")],
    labels=["0-2 km", "2-5 km", "5-10 km", "10+ km"],
    include_lowest=True
)

distance_insights = (
    predictions.groupby(
        "distance_band",
        observed=False
    )["actual_delivery_time"]
    .agg(["count", "mean", "median"])
    .round(2)
)

print(distance_insights)


# ------------------------------
# Delivery Time by Preparation Time
# ------------------------------

print("\nDelivery Time by Preparation Time")
print("---------------------------------")

predictions["preparation_band"] = pd.cut(
    predictions["preparation_time_min"],
    bins=[0, 15, 30, 45, float("inf")],
    labels=["0-15 min", "15-30 min", "30-45 min", "45+ min"],
    include_lowest=True
)

preparation_insights = (
    predictions.groupby(
        "preparation_band",
        observed=False
    )["actual_delivery_time"]
    .agg(["count", "mean", "median"])
    .round(2)
)

print(preparation_insights)

# ------------------------------
# Business Recommendations
# ------------------------------

print("\nBusiness Recommendations")
print("------------------------")

print(
    "\n1. Traffic-aware delivery planning:"
    "\n   High-traffic conditions are associated with substantially longer "
    "delivery times. Rider allocation and ETA estimates should account "
    "for current traffic conditions."
)

print(
    "\n2. Weather-aware ETA adjustment:"
    "\n   Rain and storm conditions are associated with longer delivery "
    "times. ETA calculations and operational planning should incorporate "
    "weather conditions."
)

print(
    "\n3. Distance-based delivery planning:"
    "\n   Deliveries over 10 km have substantially higher average delivery "
    "times than short-distance deliveries. Longer-distance orders should "
    "receive appropriate ETA adjustments and operational monitoring."
)

print(
    "\n4. Restaurant preparation monitoring:"
    "\n   Longer preparation times are strongly associated with longer "
    "overall delivery times. Restaurants with consistently high "
    "preparation times should be identified for operational improvement."
)

print(
    "\n5. Predictive ETA usage:"
    "\n   The delivery-time model has a mean absolute error of approximately "
    f"{mean_absolute_error:.2f} minutes with almost no systematic prediction "
    "bias. The predictions can therefore support more realistic delivery "
    "time estimates."
)