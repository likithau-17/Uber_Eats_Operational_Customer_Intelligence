from pathlib import Path

import pandas as pd

from prophet import Prophet


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

ORDERS_PATH = PROJECT_ROOT / "data" / "raw" / "orders.csv"


# Load orders
orders = pd.read_csv(ORDERS_PATH)

print("Orders Dataset")
print("=" * 50)

print(f"Shape: {orders.shape}")

print("\nColumns:")
print(orders.columns.tolist())

print("\nMissing Values:")
print(orders.isnull().sum())

print(f"\nDuplicate Rows: {orders.duplicated().sum()}")

print("\nOrder Status Distribution:")
print(orders["order_status"].value_counts())

print("\nOrder Timestamp Sample:")
print(orders["order_timestamp"].head())


# Convert timestamp to datetime
orders["order_timestamp"] = pd.to_datetime(
    orders["order_timestamp"],
    errors="coerce"
)

print("\nDatetime Conversion")
print("=" * 50)

print(f"Invalid timestamps: {orders['order_timestamp'].isna().sum()}")

print(f"Minimum timestamp: {orders['order_timestamp'].min()}")
print(f"Maximum timestamp: {orders['order_timestamp'].max()}")


# Inspect order timestamps after conversion
print("\nTimestamp Data Type:")
print(orders["order_timestamp"].dtype)


# Determine demand definition
print("\nOrder Statuses:")
print(orders["order_status"].unique())


# Aggregate orders by hour
hourly_demand = (
    orders
    .dropna(subset=["order_timestamp"])
    .set_index("order_timestamp")
    .resample("h")
    .size()
    .reset_index(name="order_count")
)

print("\nHourly Demand")
print("=" * 50)

print(f"Shape: {hourly_demand.shape}")

print("\nFirst 10 Hours:")
print(hourly_demand.head(10))

print("\nLast 10 Hours:")
print(hourly_demand.tail(10))

print("\nHourly Demand Summary:")
print(hourly_demand["order_count"].describe())

print("\nMissing Hourly Timestamps:")
print(hourly_demand["order_count"].isna().sum())

# ============================================================
# Step 2 - Time-Series EDA
# ============================================================

import matplotlib.pyplot as plt


# Create time-based features
hourly_demand["hour"] = hourly_demand["order_timestamp"].dt.hour
hourly_demand["day_of_week"] = hourly_demand["order_timestamp"].dt.day_name()
hourly_demand["day_number"] = hourly_demand["order_timestamp"].dt.dayofweek
hourly_demand["is_weekend"] = hourly_demand["day_number"] >= 5


# ------------------------------------------------------------
# 1. Overall hourly demand trend
# ------------------------------------------------------------

plt.figure(figsize=(14, 6))

plt.plot(
    hourly_demand["order_timestamp"],
    hourly_demand["order_count"]
)

plt.title("Hourly Order Demand Over Time")
plt.xlabel("Date")
plt.ylabel("Order Count")
plt.xticks(rotation=45)
plt.tight_layout()

trend_path = PROJECT_ROOT / "outputs" / "figures" / "hourly_demand_trend.png"
plt.savefig(trend_path)
plt.close()

print("\nOverall Demand Trend")
print("=" * 50)
print(f"Minimum hourly demand: {hourly_demand['order_count'].min()}")
print(f"Maximum hourly demand: {hourly_demand['order_count'].max()}")
print(f"Average hourly demand: {hourly_demand['order_count'].mean():.2f}")


# ------------------------------------------------------------
# 2. Demand by hour of day
# ------------------------------------------------------------

hourly_pattern = (
    hourly_demand
    .groupby("hour")["order_count"]
    .mean()
)

print("\nAverage Demand by Hour")
print("=" * 50)
print(hourly_pattern)

peak_hour = hourly_pattern.idxmax()
peak_hour_demand = hourly_pattern.max()

print(f"\nPeak average demand hour: {peak_hour:02d}:00")
print(f"Average orders during peak hour: {peak_hour_demand:.2f}")


plt.figure(figsize=(12, 6))

plt.plot(
    hourly_pattern.index,
    hourly_pattern.values,
    marker="o"
)

plt.title("Average Order Demand by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Average Order Count")
plt.xticks(range(24))
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

hourly_path = PROJECT_ROOT / "outputs" / "figures" / "demand_by_hour.png"
plt.savefig(hourly_path)
plt.close()


# ------------------------------------------------------------
# 3. Demand by day of week
# ------------------------------------------------------------

weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

daily_pattern = (
    hourly_demand
    .groupby(["day_number", "day_of_week"])["order_count"]
    .mean()
    .reset_index()
    .sort_values("day_number")
)

print("\nAverage Demand by Day of Week")
print("=" * 50)

for _, row in daily_pattern.iterrows():
    print(
        f"{row['day_of_week']}: "
        f"{row['order_count']:.2f} orders/hour"
    )


plt.figure(figsize=(10, 6))

plt.bar(
    daily_pattern["day_of_week"],
    daily_pattern["order_count"]
)

plt.title("Average Order Demand by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Average Order Count")
plt.xticks(rotation=30)
plt.tight_layout()

weekday_path = PROJECT_ROOT / "outputs" / "figures" / "demand_by_day_of_week.png"
plt.savefig(weekday_path)
plt.close()


# ------------------------------------------------------------
# 4. Weekday vs weekend
# ------------------------------------------------------------

weekend_pattern = (
    hourly_demand
    .groupby("is_weekend")["order_count"]
    .mean()
)

weekday_demand = weekend_pattern[False]
weekend_demand = weekend_pattern[True]

print("\nWeekday vs Weekend Demand")
print("=" * 50)
print(f"Weekday average: {weekday_demand:.2f} orders/hour")
print(f"Weekend average: {weekend_demand:.2f} orders/hour")


# ------------------------------------------------------------
# 5. Peak ordering hours
# ------------------------------------------------------------

top_hours = (
    hourly_pattern
    .sort_values(ascending=False)
    .head(5)
)

print("\nTop 5 Peak Ordering Hours")
print("=" * 50)

for hour, demand in top_hours.items():
    print(f"{hour:02d}:00 - {demand:.2f} orders/hour")

# ============================================================
# Step 3 - Time-Series Decomposition
# ============================================================

from statsmodels.tsa.seasonal import seasonal_decompose


# Create a clean time series for decomposition
demand_series = (
    hourly_demand
    .set_index("order_timestamp")["order_count"]
)


# Perform seasonal decomposition
decomposition = seasonal_decompose(
    demand_series,
    model="additive",
    period=24
)


# Display decomposition
fig = decomposition.plot()

fig.set_size_inches(14, 10)
fig.suptitle(
    "Hourly Order Demand - Time-Series Decomposition",
    fontsize=14
)

fig.tight_layout()

decomposition_path = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "demand_decomposition.png"
)

fig.savefig(decomposition_path)
plt.close(fig)


# Print decomposition summary
print("\nTime-Series Decomposition")
print("=" * 50)

print("Decomposition model: Additive")
print("Seasonal period: 24 hours")
print("Components: Trend, Seasonal, Residual")

print("\nDecomposition completed successfully.")

print(f"Figure saved to: {decomposition_path}")

# ============================================================
# Step 4 - Stationarity Test
# ============================================================

from statsmodels.tsa.stattools import adfuller


# Run Augmented Dickey-Fuller test
adf_result = adfuller(demand_series)


adf_statistic = adf_result[0]
p_value = adf_result[1]
used_lags = adf_result[2]
num_observations = adf_result[3]


print("\nAugmented Dickey-Fuller Test")
print("=" * 50)

print(f"ADF Statistic: {adf_statistic:.4f}")
print(f"p-value: {p_value:.6f}")
print(f"Number of lags used: {used_lags}")
print(f"Number of observations: {num_observations}")


print("\nCritical Values:")

for key, value in adf_result[4].items():
    print(f"{key}: {value:.4f}")


# Interpret the test
print("\nStationarity Interpretation")
print("=" * 50)

if p_value < 0.05:
    print("Result: The series is stationary.")
    print("Decision: Reject the null hypothesis.")
else:
    print("Result: The series is non-stationary.")
    print("Decision: Fail to reject the null hypothesis.")

# ============================================================
# Step 5 - Chronological Train/Test Split
# ============================================================


# Remove time-based EDA columns from the forecasting series
# The original hourly demand series remains unchanged.
forecast_series = (
    hourly_demand
    .set_index("order_timestamp")["order_count"]
    .sort_index()
)


# Calculate chronological split point
train_size = int(len(forecast_series) * 0.80)

train_series = forecast_series.iloc[:train_size]
test_series = forecast_series.iloc[train_size:]


print("\nChronological Train/Test Split")
print("=" * 50)

print(f"Total observations: {len(forecast_series)}")
print(f"Training observations: {len(train_series)}")
print(f"Test observations: {len(test_series)}")

print("\nTraining Period:")
print(f"Start: {train_series.index.min()}")
print(f"End:   {train_series.index.max()}")

print("\nTest Period:")
print(f"Start: {test_series.index.min()}")
print(f"End:   {test_series.index.max()}")


# Verify chronological ordering
print("\nChronological Validation")
print("=" * 50)

if train_series.index.max() < test_series.index.min():
    print("Valid: Training data occurs entirely before test data.")
else:
    print("ERROR: Training and test periods overlap.")


# Verify no observations were lost
print("\nObservation Check")
print("=" * 50)

print(
    f"Train + Test = "
    f"{len(train_series) + len(test_series)} observations"
)

print(
    f"Original = "
    f"{len(forecast_series)} observations"
)

# ============================================================
# Step 6 - ARIMA Forecasting
# ============================================================

from statsmodels.tsa.arima.model import ARIMA


# Build ARIMA model
arima_model = ARIMA(
    train_series,
    order=(1, 0, 1)
)


# Fit model
arima_result = arima_model.fit()


print("\nARIMA Model")
print("=" * 50)

print("Model: ARIMA(1, 0, 1)")
print("Training completed successfully.")

print("\nModel Summary:")
print(arima_result.summary())


# Forecast the complete test period
arima_forecast = arima_result.forecast(
    steps=len(test_series)
)


# Align forecast index with test data
arima_forecast.index = test_series.index


print("\nARIMA Forecast")
print("=" * 50)

print(f"Forecast observations: {len(arima_forecast)}")

print("\nFirst 10 Forecasts:")
print(arima_forecast.head(10))

print("\nLast 10 Forecasts:")
print(arima_forecast.tail(10))

# ============================================================
# Step 6B - ARIMA Evaluation
# ============================================================

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


# Calculate evaluation metrics
mae = mean_absolute_error(
    test_series,
    arima_forecast
)

rmse = np.sqrt(
    mean_squared_error(
        test_series,
        arima_forecast
    )
)


# MAPE excluding zero-demand observations
non_zero_actuals = test_series != 0

mape = (
    np.mean(
        np.abs(
            (
                test_series[non_zero_actuals]
                - arima_forecast[non_zero_actuals]
            )
            / test_series[non_zero_actuals]
        )
    )
    * 100
)


print("\nARIMA Evaluation")
print("=" * 50)

print(f"MAE:  {mae:.2f} orders")
print(f"RMSE: {rmse:.2f} orders")
print(f"MAPE: {mape:.2f}%")

print(
    f"\nMAPE calculated using "
    f"{non_zero_actuals.sum()} non-zero-demand observations "
    f"out of {len(test_series)} total test observations."
)

# ============================================================
# Step 7 - Prophet Forecasting
# ============================================================

# Prepare training data for Prophet
prophet_train = (
    train_series
    .reset_index()
    .rename(
        columns={
            "order_timestamp": "ds",
            "order_count": "y"
        }
    )
)


# Create Prophet model
prophet_model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=False
)


# Fit Prophet model
prophet_model.fit(prophet_train)


print("\nProphet Model")
print("=" * 50)

print("Daily seasonality: Enabled")
print("Weekly seasonality: Enabled")
print("Yearly seasonality: Disabled")
print("Training completed successfully.")


# Create future dataframe for the test period
future = pd.DataFrame({
    "ds": test_series.index
})


# Generate forecast
prophet_prediction = prophet_model.predict(future)


# Extract predicted demand
prophet_forecast = (
    prophet_prediction
    .set_index("ds")["yhat"]
)


print("\nProphet Forecast")
print("=" * 50)

print(f"Forecast observations: {len(prophet_forecast)}")

print("\nFirst 10 Forecasts:")
print(prophet_forecast.head(10))

print("\nLast 10 Forecasts:")
print(prophet_forecast.tail(10))

# ============================================================
# Step 7B - Prophet Evaluation
# ============================================================

# Calculate evaluation metrics
prophet_mae = mean_absolute_error(
    test_series,
    prophet_forecast
)

prophet_rmse = np.sqrt(
    mean_squared_error(
        test_series,
        prophet_forecast
    )
)


# MAPE excluding zero-demand observations
non_zero_actuals = test_series != 0

prophet_mape = (
    np.mean(
        np.abs(
            (
                test_series[non_zero_actuals]
                - prophet_forecast[non_zero_actuals]
            )
            / test_series[non_zero_actuals]
        )
    )
    * 100
)


print("\nProphet Evaluation")
print("=" * 50)

print(f"MAE:  {prophet_mae:.2f} orders")
print(f"RMSE: {prophet_rmse:.2f} orders")
print(f"MAPE: {prophet_mape:.2f}%")

print(
    f"\nMAPE calculated using "
    f"{non_zero_actuals.sum()} non-zero-demand observations "
    f"out of {len(test_series)} total test observations."
)

# ============================================================
# Step 8 - ARIMA vs Prophet Comparison
# ============================================================

model_comparison = pd.DataFrame({
    "Model": [
        "ARIMA(1,0,1)",
        "Prophet"
    ],
    "MAE": [
        mae,
        prophet_mae
    ],
    "RMSE": [
        rmse,
        prophet_rmse
    ],
    "MAPE": [
        mape,
        prophet_mape
    ]
})


print("\nModel Comparison")
print("=" * 70)

print(
    model_comparison.to_string(
        index=False,
        formatters={
            "MAE": "{:.2f}".format,
            "RMSE": "{:.2f}".format,
            "MAPE": "{:.2f}%".format
        }
    )
)


# Determine the best model based on each metric
best_mae_model = model_comparison.loc[
    model_comparison["MAE"].idxmin(),
    "Model"
]

best_rmse_model = model_comparison.loc[
    model_comparison["RMSE"].idxmin(),
    "Model"
]

best_mape_model = model_comparison.loc[
    model_comparison["MAPE"].idxmin(),
    "Model"
]


print("\nBest Model by Metric")
print("=" * 50)

print(f"Lowest MAE:  {best_mae_model}")
print(f"Lowest RMSE: {best_rmse_model}")
print(f"Lowest MAPE: {best_mape_model}")


# Select final model
if (
    best_mae_model == best_rmse_model
    and best_rmse_model == best_mape_model
):
    selected_model = best_mae_model
else:
    selected_model = (
        model_comparison
        .sort_values(["MAE", "RMSE", "MAPE"])
        .iloc[0]["Model"]
    )


print("\nSelected Forecasting Model")
print("=" * 50)
print(f"Selected model: {selected_model}")

# ============================================================
# Step 9 - Final Forecast Output
# ============================================================

# Create final forecast results
forecast_output = pd.DataFrame({
    "timestamp": test_series.index,
    "actual_demand": test_series.values,
    "forecasted_demand": prophet_forecast.values
})


# Calculate forecast error
forecast_output["error"] = (
    forecast_output["actual_demand"]
    - forecast_output["forecasted_demand"]
)


# Save forecast predictions
predictions_dir = PROJECT_ROOT / "outputs" / "predictions"
predictions_dir.mkdir(parents=True, exist_ok=True)

forecast_path = (
    predictions_dir
    / "hourly_demand_forecast.csv"
)

forecast_output.to_csv(
    forecast_path,
    index=False
)


print("\nFinal Forecast Output")
print("=" * 50)

print(f"Forecast rows: {len(forecast_output)}")

print("\nFirst 10 Forecast Results:")
print(
    forecast_output
    .head(10)
    .to_string(index=False)
)

print(f"\nForecast saved to: {forecast_path}")


# ------------------------------------------------------------
# Forecast Visualization
# ------------------------------------------------------------

plt.figure(figsize=(16, 7))

plt.plot(
    train_series.index,
    train_series.values,
    label="Training Demand"
)

plt.plot(
    test_series.index,
    test_series.values,
    label="Actual Test Demand"
)

plt.plot(
    prophet_forecast.index,
    prophet_forecast.values,
    label="Prophet Forecast"
)

plt.title("Hourly Demand Forecast - Prophet")
plt.xlabel("Date")
plt.ylabel("Order Count")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()


forecast_figure_path = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "hourly_demand_forecast.png"
)

plt.savefig(forecast_figure_path)
plt.close()


print(f"Forecast figure saved to: {forecast_figure_path}")