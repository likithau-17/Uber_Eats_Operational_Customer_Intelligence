from pathlib import Path

import pandas as pd


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"


# ============================================================
# Input Files
# ============================================================

CUSTOMER_SEGMENTS_PATH = (
    PROCESSED_DATA_DIR / "customer_segments.csv"
)

NEGATIVE_THEMES_PATH = (
    PROCESSED_DATA_DIR / "negative_review_themes.csv"
)

SENTIMENT_PREDICTIONS_PATH = (
    PREDICTIONS_DIR / "sentiment_predictions.csv"
)

DELIVERY_PREDICTIONS_PATH = (
    PREDICTIONS_DIR / "delivery_time_predictions.csv"
)

DEMAND_FORECAST_PATH = (
    PREDICTIONS_DIR / "hourly_demand_forecast.csv"
)


# ============================================================
# Load Existing Module Outputs
# ============================================================

customer_segments = pd.read_csv(
    CUSTOMER_SEGMENTS_PATH
)

negative_review_themes = pd.read_csv(
    NEGATIVE_THEMES_PATH
)

sentiment_predictions = pd.read_csv(
    SENTIMENT_PREDICTIONS_PATH
)

delivery_predictions = pd.read_csv(
    DELIVERY_PREDICTIONS_PATH
)

demand_forecast = pd.read_csv(
    DEMAND_FORECAST_PATH
)


# ============================================================
# 1. Customer Segmentation Business Analysis
# ============================================================

segment_analysis = (
    customer_segments
    .groupby("segment_name")
    .agg(
        customer_count=("customer_id", "count"),
        avg_total_orders=("total_orders", "mean"),
        avg_order_value=("avg_order_value", "mean"),
        avg_total_spending=("total_spending", "mean"),
        avg_ordering_frequency=("ordering_frequency", "mean"),
        avg_weekend_orders=("weekend_orders", "mean"),
        avg_late_night_orders=("late_night_orders", "mean"),
    )
    .sort_values(
        "avg_total_spending",
        ascending=False
    )
    .round(2)
)


print("\nCUSTOMER SEGMENTATION INSIGHTS")
print("=" * 60)

print("\n1. Segment Profile")
print("------------------")

print(
    segment_analysis.to_string()
)


highest_value_segment = (
    segment_analysis["avg_total_spending"]
    .idxmax()
)

highest_value_spending = (
    segment_analysis.loc[
        highest_value_segment,
        "avg_total_spending"
    ]
)


print("\n2. Highest-Value Segment")
print("------------------------")

print(
    f"Segment: {highest_value_segment}"
)

print(
    f"Average total spending per customer: "
    f"{highest_value_spending:.2f}"
)


# ============================================================
# 2. Customer Experience / NLP Business Analysis
# ============================================================

sentiment_distribution = (
    sentiment_predictions["sentiment"]
    .value_counts()
)

sentiment_percentage = (
    sentiment_predictions["sentiment"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


print("\nCUSTOMER EXPERIENCE INSIGHTS")
print("=" * 60)


print("\n1. Overall Sentiment Distribution")
print("----------------------------------")

for sentiment, count in sentiment_distribution.items():

    percentage = sentiment_percentage[sentiment]

    print(
        f"{sentiment}: "
        f"{count} reviews "
        f"({percentage:.2f}%)"
    )


negative_review_count = (
    sentiment_distribution.get("Negative", 0)
)

negative_review_percentage = (
    sentiment_percentage.get("Negative", 0)
)


print("\n2. Negative Review Summary")
print("---------------------------")

print(
    f"Negative reviews: "
    f"{negative_review_count}"
)

print(
    f"Percentage of reviews that are negative: "
    f"{negative_review_percentage:.2f}%"
)


print("\n3. Negative Review Themes")
print("-------------------------")

print(
    negative_review_themes
    .sort_values(
        "percentage_of_negative_reviews",
        ascending=False
    )
    .to_string(index=False)
)


top_complaint_theme = (
    negative_review_themes
    .sort_values(
        "percentage_of_negative_reviews",
        ascending=False
    )
    .iloc[0]
)


print("\n4. Most Common Negative Theme")
print("-----------------------------")

print(
    f"Theme: "
    f"{top_complaint_theme['theme']}"
)

print(
    f"Share of negative reviews: "
    f"{top_complaint_theme['percentage_of_negative_reviews']:.2f}%"
)


# ============================================================
# 3. Delivery Prediction Business Analysis
# ============================================================

delivery_predictions["prediction_error"] = (
    delivery_predictions["predicted_delivery_time"]
    - delivery_predictions["actual_delivery_time"]
)

delivery_predictions["absolute_error"] = (
    delivery_predictions["prediction_error"].abs()
)

delivery_mae = (
    delivery_predictions["absolute_error"].mean()
)

delivery_mean_error = (
    delivery_predictions["prediction_error"].mean()
)


print("\nDELIVERY OPERATIONS INSIGHTS")
print("=" * 60)


print("\n1. Prediction Performance")
print("-------------------------")

print(
    f"Mean Absolute Error: "
    f"{delivery_mae:.2f} minutes"
)

print(
    f"Mean Prediction Error: "
    f"{delivery_mean_error:.2f} minutes"
)


# ------------------------------------------------------------
# Delivery Time by Traffic
# ------------------------------------------------------------

traffic_insights = (
    delivery_predictions
    .groupby("traffic_condition")[
        "actual_delivery_time"
    ]
    .agg(["count", "mean", "median"])
    .sort_values(
        "mean",
        ascending=False
    )
    .round(2)
)


print("\n2. Delivery Time by Traffic Condition")
print("--------------------------------------")

print(
    traffic_insights.to_string()
)


# ------------------------------------------------------------
# Delivery Time by Weather
# ------------------------------------------------------------

weather_insights = (
    delivery_predictions
    .groupby("weather")[
        "actual_delivery_time"
    ]
    .agg(["count", "mean", "median"])
    .sort_values(
        "mean",
        ascending=False
    )
    .round(2)
)


print("\n3. Delivery Time by Weather")
print("---------------------------")

print(
    weather_insights.to_string()
)


# ------------------------------------------------------------
# Delivery Time by Distance
# ------------------------------------------------------------

delivery_predictions["distance_band"] = pd.cut(
    delivery_predictions["delivery_distance_km"],
    bins=[
        0,
        2,
        5,
        10,
        float("inf")
    ],
    labels=[
        "0-2 km",
        "2-5 km",
        "5-10 km",
        "10+ km"
    ],
    include_lowest=True
)


distance_insights = (
    delivery_predictions
    .groupby(
        "distance_band",
        observed=False
    )["actual_delivery_time"]
    .agg(["count", "mean", "median"])
    .round(2)
)


print("\n4. Delivery Time by Distance")
print("----------------------------")

print(
    distance_insights.to_string()
)


# ------------------------------------------------------------
# Delivery Time by Preparation Time
# ------------------------------------------------------------

delivery_predictions["preparation_band"] = pd.cut(
    delivery_predictions["preparation_time_min"],
    bins=[
        0,
        15,
        30,
        45,
        float("inf")
    ],
    labels=[
        "0-15 min",
        "15-30 min",
        "30-45 min",
        "45+ min"
    ],
    include_lowest=True
)


preparation_insights = (
    delivery_predictions
    .groupby(
        "preparation_band",
        observed=False
    )["actual_delivery_time"]
    .agg(["count", "mean", "median"])
    .round(2)
)


print("\n5. Delivery Time by Preparation Time")
print("-------------------------------------")

print(
    preparation_insights.to_string()
)


# ------------------------------------------------------------
# Largest Observed Delivery-Time Differences
# ------------------------------------------------------------

traffic_range = (
    traffic_insights["mean"].max()
    - traffic_insights["mean"].min()
)

weather_range = (
    weather_insights["mean"].max()
    - weather_insights["mean"].min()
)

distance_range = (
    distance_insights["mean"].max()
    - distance_insights["mean"].min()
)

preparation_range = (
    preparation_insights["mean"].max()
    - preparation_insights["mean"].min()
)


print("\n6. Observed Delivery-Time Differences")
print("--------------------------------------")

print(
    f"Traffic range: "
    f"{traffic_range:.2f} minutes"
)

print(
    f"Weather range: "
    f"{weather_range:.2f} minutes"
)

print(
    f"Distance range: "
    f"{distance_range:.2f} minutes"
)

print(
    f"Preparation-time range: "
    f"{preparation_range:.2f} minutes"
)


# ============================================================
# 4. Demand Forecasting Business Analysis
# ============================================================

demand_forecast["timestamp"] = pd.to_datetime(
    demand_forecast["timestamp"]
)


# ------------------------------------------------------------
# Create Time Features
# ------------------------------------------------------------

demand_forecast["hour"] = (
    demand_forecast["timestamp"].dt.hour
)

demand_forecast["day_of_week"] = (
    demand_forecast["timestamp"].dt.dayofweek
)

demand_forecast["is_weekend"] = (
    demand_forecast["day_of_week"] >= 5
)


# ------------------------------------------------------------
# Forecast Performance
# ------------------------------------------------------------

demand_forecast["absolute_error"] = (
    demand_forecast["actual_demand"]
    - demand_forecast["forecasted_demand"]
).abs()

demand_forecast["squared_error"] = (
    demand_forecast["actual_demand"]
    - demand_forecast["forecasted_demand"]
) ** 2


forecast_mae = (
    demand_forecast["absolute_error"].mean()
)

forecast_rmse = (
    demand_forecast["squared_error"].mean()
) ** 0.5


# ------------------------------------------------------------
# MAPE
# ------------------------------------------------------------

non_zero_demand = (
    demand_forecast["actual_demand"] != 0
)

forecast_mape = (
    (
        (
            demand_forecast.loc[
                non_zero_demand,
                "actual_demand"
            ]
            - demand_forecast.loc[
                non_zero_demand,
                "forecasted_demand"
            ]
        ).abs()
        / demand_forecast.loc[
            non_zero_demand,
            "actual_demand"
        ]
    ).mean()
    * 100
)


# ------------------------------------------------------------
# Demand Summary
# ------------------------------------------------------------

average_actual_demand = (
    demand_forecast["actual_demand"].mean()
)

average_forecasted_demand = (
    demand_forecast["forecasted_demand"].mean()
)

average_forecast_error = (
    demand_forecast["error"].mean()
)


# ------------------------------------------------------------
# Peak Demand
# ------------------------------------------------------------

peak_row = demand_forecast.loc[
    demand_forecast["actual_demand"].idxmax()
]

peak_timestamp = peak_row["timestamp"]

peak_actual_demand = (
    peak_row["actual_demand"]
)

peak_forecasted_demand = (
    peak_row["forecasted_demand"]
)


# ------------------------------------------------------------
# Average Demand by Hour
# ------------------------------------------------------------

hourly_demand = (
    demand_forecast
    .groupby("hour")["actual_demand"]
    .mean()
    .sort_values(ascending=False)
)

peak_hour = hourly_demand.idxmax()

peak_hour_demand = hourly_demand.max()


# ------------------------------------------------------------
# Weekend vs Weekday Demand
# ------------------------------------------------------------

weekday_demand = (
    demand_forecast.loc[
        ~demand_forecast["is_weekend"],
        "actual_demand"
    ].mean()
)

weekend_demand = (
    demand_forecast.loc[
        demand_forecast["is_weekend"],
        "actual_demand"
    ].mean()
)


print("\nDEMAND FORECASTING INSIGHTS")
print("=" * 60)


print("\n1. Forecast Performance")
print("-----------------------")

print(
    f"MAE: "
    f"{forecast_mae:.2f} orders"
)

print(
    f"RMSE: "
    f"{forecast_rmse:.2f} orders"
)

print(
    f"MAPE: "
    f"{forecast_mape:.2f}%"
)


print("\n2. Forecast Period Demand Summary")
print("----------------------------------")

print(
    f"Average actual demand: "
    f"{average_actual_demand:.2f} orders/hour"
)

print(
    f"Average forecasted demand: "
    f"{average_forecasted_demand:.2f} orders/hour"
)

print(
    f"Average forecast error: "
    f"{average_forecast_error:.2f} orders/hour"
)


print("\n3. Peak Demand in Forecast Period")
print("----------------------------------")

print(
    f"Peak timestamp: "
    f"{peak_timestamp}"
)

print(
    f"Actual demand at peak: "
    f"{peak_actual_demand:.0f} orders/hour"
)

print(
    f"Forecasted demand at peak: "
    f"{peak_forecasted_demand:.2f} orders/hour"
)


print("\n4. Peak Hour Analysis")
print("---------------------")

print(
    f"Highest average demand hour: "
    f"{peak_hour:02d}:00"
)

print(
    f"Average demand at this hour: "
    f"{peak_hour_demand:.2f} orders/hour"
)


print("\n5. Weekend vs Weekday Demand")
print("----------------------------")

print(
    f"Weekday average demand: "
    f"{weekday_demand:.2f} orders/hour"
)

print(
    f"Weekend average demand: "
    f"{weekend_demand:.2f} orders/hour"
)


print("\n6. Forecast Bias Check")
print("----------------------")

if average_forecast_error > 0:

    print(
        "Result: The forecast usually predicts "
        "slightly less demand than actually occurs."
    )

elif average_forecast_error < 0:

    print(
        "Result: The forecast usually predicts "
        "slightly more demand than actually occurs."
    )

else:

    print(
        "Result: The forecast does not show "
        "a clear average bias."
    )


# ============================================================
# 5. Cross-Module Business Insights
# ============================================================

print("\nCROSS-MODULE BUSINESS INSIGHTS")
print("=" * 60)


# ------------------------------------------------------------
# 1. Customer Value
# ------------------------------------------------------------

high_value_segment = segment_analysis.loc[
    segment_analysis["avg_total_spending"].idxmax()
]

low_value_segment = segment_analysis.loc[
    segment_analysis["avg_total_spending"].idxmin()
]

spending_difference = (
    high_value_segment["avg_total_spending"]
    - low_value_segment["avg_total_spending"]
)


print("\n1. Customer Value")
print("-----------------")

print(
    f"The {high_value_segment.name} segment spends the "
    f"most on average: "
    f"{high_value_segment['avg_total_spending']:.2f} "
    f"per customer."
)

print(
    f"This is {spending_difference:.2f} more than the "
    f"{low_value_segment.name} segment."
)

print(
    f"There are "
    f"{int(high_value_segment['customer_count'])} "
    f"customers in the highest-value segment."
)


# ------------------------------------------------------------
# 2. Customer Experience
# ------------------------------------------------------------

print("\n2. Customer Experience")
print("----------------------")

print(
    f"{negative_review_percentage:.2f}% of the evaluated "
    f"reviews are negative."
)

print(
    f"The most common negative theme is "
    f"'{top_complaint_theme['theme']}'."
)

print(
    f"It represents "
    f"{top_complaint_theme['percentage_of_negative_reviews']:.2f}% "
    f"of categorized negative reviews."
)


# ------------------------------------------------------------
# 3. Delivery Operations
# ------------------------------------------------------------

print("\n3. Delivery Operations")
print("----------------------")

print(
    f"Distance has the largest difference in average "
    f"delivery time among the analyzed factors: "
    f"{distance_range:.2f} minutes."
)

print(
    f"Preparation time has the second-largest difference: "
    f"{preparation_range:.2f} minutes."
)

if "High" in traffic_insights.index and "Low" in traffic_insights.index:

    high_traffic_time = traffic_insights.loc[
        "High",
        "mean"
    ]

    low_traffic_time = traffic_insights.loc[
        "Low",
        "mean"
    ]

    print(
        f"High-traffic deliveries take about "
        f"{high_traffic_time:.2f} minutes on average, "
        f"compared with "
        f"{low_traffic_time:.2f} minutes during low traffic."
    )


# ------------------------------------------------------------
# 4. Demand and Capacity
# ------------------------------------------------------------

print("\n4. Demand and Capacity")
print("----------------------")

print(
    f"The highest average demand occurs at "
    f"{peak_hour:02d}:00, with approximately "
    f"{peak_hour_demand:.2f} orders/hour."
)

print(
    f"Weekend demand averages "
    f"{weekend_demand:.2f} orders/hour, compared with "
    f"{weekday_demand:.2f} orders/hour on weekdays."
)

print(
    "This means evening and weekend periods need "
    "more attention when planning delivery capacity."
)


# ------------------------------------------------------------
# 5. Forecasting
# ------------------------------------------------------------

print("\n5. Forecasting")
print("--------------")

print(
    f"The demand forecasting model has an MAE of "
    f"{forecast_mae:.2f} orders/hour."
)

print(
    f"Its RMSE is "
    f"{forecast_rmse:.2f} orders/hour."
)

print(
    f"The average forecast error is "
    f"{average_forecast_error:.2f} orders/hour."
)

print(
    "The forecast is useful for planning, but individual "
    "demand spikes may still be difficult to predict."
)


# ============================================================
# 6. Business Recommendations
# ============================================================

print("\nBUSINESS RECOMMENDATIONS")
print("=" * 60)


# ------------------------------------------------------------
# 1. Customer Retention
# ------------------------------------------------------------

print("\n1. Retain High-Value Customers")
print("------------------------------")

print(
    f"The {high_value_segment.name} segment has the "
    f"highest average spending at "
    f"{high_value_segment['avg_total_spending']:.2f} "
    f"per customer."
)

print(
    "Recommendation: Give this group special attention "
    "through loyalty rewards, personalized offers, and "
    "other retention strategies."
)


# ------------------------------------------------------------
# 2. Delivery Delay Reduction
# ------------------------------------------------------------

print("\n2. Reduce Delivery Delays")
print("-------------------------")

print(
    f"Longer delivery distances show the largest difference "
    f"in delivery time, followed by restaurant preparation time."
)

print(
    "Recommendation: Monitor long-distance orders and "
    "restaurants with long preparation times more closely. "
    "Use traffic information to improve delivery planning "
    "and estimated delivery times."
)


# ------------------------------------------------------------
# 3. Peak Capacity Planning
# ------------------------------------------------------------

print("\n3. Plan More Capacity During Peak Hours")
print("----------------------------------------")

print(
    f"The highest average demand occurs around "
    f"{peak_hour:02d}:00."
)

print(
    f"Weekend demand is "
    f"{weekend_demand:.2f} orders/hour compared with "
    f"{weekday_demand:.2f} orders/hour on weekdays."
)

print(
    "Recommendation: Increase rider availability and "
    "operational capacity during high-demand evening "
    "and weekend periods."
)


# ------------------------------------------------------------
# 4. Customer Experience Improvement
# ------------------------------------------------------------

print("\n4. Reduce the Main Customer Complaint")
print("--------------------------------------")

print(
    f"The most common negative-review theme is "
    f"'{top_complaint_theme['theme']}'."
)

print(
    f"It accounts for "
    f"{top_complaint_theme['percentage_of_negative_reviews']:.2f}% "
    f"of categorized negative reviews."
)

print(
    "Recommendation: Investigate the operational causes "
    "behind this complaint and improve the areas that "
    "directly affect the customer experience."
)


# ------------------------------------------------------------
# 5. Use Forecasts for Planning
# ------------------------------------------------------------

print("\n5. Use Demand Forecasts for Planning")
print("------------------------------------")

print(
    f"The forecasting model has an MAE of "
    f"{forecast_mae:.2f} orders/hour."
)

print(
    "Recommendation: Use the forecast to plan rider "
    "availability and operational resources, but continue "
    "monitoring actual demand because sudden spikes may "
    "not be predicted perfectly."
)


# ============================================================
# 7. Model Limitations
# ============================================================

print("\nMODEL LIMITATIONS")
print("=" * 60)


print("\n1. Synthetic Data")
print("------------------")

print(
    "The project uses synthetic data rather than real Uber Eats "
    "production data. Therefore, the patterns and model results "
    "may not represent real-world business performance."
)


print("\n2. NLP Model Performance")
print("------------------------")

print(
    "The sentiment model achieved extremely high accuracy on "
    "the synthetic reviews. This is likely because the generated "
    "reviews contain repetitive and predictable language."
)

print(
    "Therefore, the NLP accuracy should not be treated as "
    "evidence that the model would perform equally well on "
    "real customer reviews."
)


print("\n3. Limited Business Variables")
print("-----------------------------")

print(
    "The analysis uses a limited set of operational and "
    "customer variables. Real delivery performance can also "
    "depend on factors such as location, rider availability, "
    "restaurant workload, and real-time events."
)


print("\n4. Forecasting Limitations")
print("--------------------------")

print(
    "The demand forecasting model can capture historical "
    "patterns, but unexpected events can cause large changes "
    "in demand that the model may not predict."
)


print("\n5. Model Predictions Are Not Perfect")
print("-------------------------------------")

print(
    "Delivery-time and demand predictions contain errors. "
    "The models should therefore support business decisions "
    "rather than completely replace human judgment."
)


# ============================================================
# 8. Future Improvement Opportunities
# ============================================================

print("\nFUTURE IMPROVEMENT OPPORTUNITIES")
print("=" * 60)


print("\n1. Use Real Production Data")
print("---------------------------")

print(
    "Replace synthetic data with real historical delivery, "
    "customer, restaurant, rider, and review data."
)


print("\n2. Add More Operational Features")
print("--------------------------------")

print(
    "Add features such as delivery zone, rider availability, "
    "restaurant workload, order size, special events, and "
    "real-time traffic conditions."
)


print("\n3. Improve the NLP Model")
print("------------------------")

print(
    "Train and test the sentiment model on real customer reviews "
    "and experiment with more advanced language models such as "
    "BERT or other transformer-based models."
)


print("\n4. Improve Delivery-Time Prediction")
print("------------------------------------")

print(
    "Test additional machine-learning models and use stronger "
    "feature engineering to improve delivery-time accuracy."
)


print("\n5. Improve Demand Forecasting")
print("-----------------------------")

print(
    "Add more historical data, holidays, special events, "
    "weather, and location-level demand to improve forecasting "
    "during unusual demand periods."
)


print("\n6. Build a Production Pipeline")
print("------------------------------")

print(
    "Automate the process so that new data can move through "
    "cleaning, feature engineering, prediction, and reporting "
    "without manually running each step."
)


print("\n7. Monitor Models Over Time")
print("---------------------------")

print(
    "Track model performance after deployment and retrain "
    "the models when customer behavior, demand patterns, "
    "or delivery conditions change."
)