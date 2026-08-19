# Hourly Demand Forecasting

## 1. Objective

The objective of this module is to forecast hourly Uber Eats order demand using time-series forecasting techniques.

The forecasting target is:

**Hourly order volume**

The analysis compares two forecasting approaches:

- ARIMA
- Prophet

The models are evaluated using a chronological train/test split and the following metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

The final forecasting model is selected based on its performance on the held-out test period.

---

## 2. Dataset

The primary dataset used for demand forecasting is:

`data/raw/orders.csv`

The dataset contains:

- **50,000 orders**
- **11 columns**
- Order information
- Customer and restaurant identifiers
- Order timestamp
- Order amount
- Number of items
- Order status
- City
- Weather
- Traffic condition

The order data covers the period:

**January 1, 2026 → June 30, 2026**

### Order Status

The dataset contains:

| Order Status | Count |
|---|---:|
| Completed | 46,046 |
| Cancelled | 3,954 |

---

## 3. Demand Definition

For this forecasting module, demand is defined as:

> **The number of orders placed during each hour, regardless of order status.**

Cancelled orders are included because the objective is to measure **customer ordering demand**, rather than completed delivery volume.

This distinction is important:

- **Demand forecasting:** all orders placed
- **Delivery-time prediction:** completed/delivered orders

Therefore, the forecasting target represents the level of customer demand entering the marketplace.

---

## 4. Data Preparation

The `order_timestamp` column was converted from string format to a datetime data type.

Validation confirmed:

- No missing timestamps
- No invalid timestamps
- No duplicate rows
- Valid timestamp range
- Complete hourly time index after aggregation

The timestamp range was:

**2026-01-01 00:00:00 → 2026-06-30 23:00:00**

Orders were aggregated into hourly intervals.

The resulting hourly time series contained:

**4,344 hourly observations**

The resulting structure was:

| timestamp | order_count |
|---|---:|
| 2026-01-01 00:00:00 | 13 |
| 2026-01-01 01:00:00 | 8 |
| 2026-01-01 02:00:00 | 8 |

Zero-demand hours were retained because they are valid observations in a continuous hourly time series.

---

## 5. Time-Series EDA

The hourly demand series had the following overall characteristics:

| Statistic | Value |
|---|---:|
| Minimum hourly demand | 0 |
| Maximum hourly demand | 45 |
| Mean hourly demand | 11.51 |
| Standard deviation | 6.87 |

The EDA showed clear recurring demand patterns.

### 5.1 Hour-of-Day Pattern

Demand was relatively low during the early morning and increased during lunch and evening periods.

The strongest demand occurred during the evening.

The top five hours by average demand were:

| Hour | Average orders/hour |
|---|---:|
| 22:00 | 21.62 |
| 21:00 | 21.13 |
| 20:00 | 20.82 |
| 19:00 | 20.73 |
| 18:00 | 20.45 |

The peak average demand occurred at:

**22:00 — 21.62 orders/hour**

This demonstrates strong daily seasonality.

---

### 5.2 Day-of-Week Pattern

Average hourly demand by day was:

| Day | Average orders/hour |
|---|---:|
| Monday | 9.96 |
| Tuesday | 10.16 |
| Wednesday | 9.78 |
| Thursday | 10.26 |
| Friday | 10.25 |
| Saturday | 15.21 |
| Sunday | 14.87 |

Saturday and Sunday had substantially higher demand than the other days.

---

### 5.3 Weekday vs Weekend

Average demand was:

- **Weekday:** 10.09 orders/hour
- **Weekend:** 15.04 orders/hour

Weekend demand was approximately **49% higher** than weekday demand.

This indicates strong weekly seasonality.

---

## 6. Time-Series Decomposition

The hourly demand series was decomposed into:

- Observed
- Trend
- Seasonal
- Residual

An **additive decomposition** was used.

The seasonal period was set to:

**24 hours**

This was selected because the data is hourly and the EDA identified a strong repeating hour-of-day pattern.

The decomposition helped separate the demand series into its underlying trend, recurring daily seasonal component, and residual variation.

Output:

`outputs/figures/demand_decomposition.png`

---

## 7. Stationarity Testing

The **Augmented Dickey-Fuller (ADF) test** was performed on the original hourly demand series.

### ADF Results

| Statistic | Value |
|---|---:|
| ADF Statistic | -9.2150 |
| p-value | 0.000000 |
| Lags Used | 31 |
| Observations | 4,312 |

Critical values:

| Significance Level | Critical Value |
|---|---:|
| 1% | -3.4319 |
| 5% | -2.8622 |
| 10% | -2.5671 |

The p-value was below 0.05.

Therefore:

> The null hypothesis of a unit root was rejected.

The series was considered stationary based on the ADF test.

### Differencing

Because the original series was found to be stationary, differencing was not applied at this stage.

Therefore, the ARIMA models were evaluated beginning with:

**d = 0**

---

## 8. Chronological Train/Test Split

Random train/test splitting was not used because time-series forecasting must preserve temporal ordering.

Random splitting could introduce temporal leakage by allowing information from later periods to appear in the training data.

An **80/20 chronological split** was used.

### Training Data

**3,475 observations**

Period:

**2026-01-01 00:00 → 2026-05-25 18:00**

### Test Data

**869 observations**

Period:

**2026-05-25 19:00 → 2026-06-30 23:00**

The training period occurred entirely before the test period.

The same test period was used for both ARIMA and Prophet to ensure a fair comparison.

---

## 9. ARIMA Forecasting

An ARIMA model was developed as the first forecasting approach.

The initial configuration was:

**ARIMA(1,0,1)**

The components are:

- **p = 1:** one autoregressive term
- **d = 0:** no differencing
- **q = 1:** one moving-average term

The choice of `d = 0` was supported by the ADF stationarity test.

The model was trained only on the chronological training period and used to forecast the 869-hour test period.

### ARIMA Results

| Metric | Result |
|---|---:|
| MAE | 5.54 orders |
| RMSE | 7.01 orders |
| MAPE | 74.18% |

The ARIMA forecast tended to converge toward the overall demand level rather than reproducing the strong hourly seasonal pattern.

This limitation is consistent with the use of a simple non-seasonal ARIMA model.

---

## 10. Prophet Forecasting

Prophet was used as the second forecasting approach.

The model was configured with:

- Daily seasonality: **Enabled**
- Weekly seasonality: **Enabled**
- Yearly seasonality: **Disabled**

Yearly seasonality was not enabled because the dataset contains only six months of historical observations, which is insufficient to reliably estimate an annual seasonal pattern.

Prophet was trained using the same training period used for ARIMA.

The model then generated forecasts for the same 869-hour test period.

### Prophet Results

| Metric | Result |
|---|---:|
| MAE | 3.11 orders |
| RMSE | 4.10 orders |
| MAPE | 38.86% |

---

## 11. ARIMA vs Prophet

The two forecasting models were compared using the same test period.

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| ARIMA(1,0,1) | 5.54 | 7.01 | 74.18% |
| **Prophet** | **3.11** | **4.10** | **38.86%** |

Lower values indicate better forecasting performance.

Prophet achieved the lowest:

- MAE
- RMSE
- MAPE

Therefore, **Prophet was selected as the final forecasting model**.

The improvement is consistent with the strong daily and weekly seasonality identified during exploratory analysis.

---

## 12. Final Forecast

The final forecast was generated using the selected Prophet model.

The forecast output contains:

- Timestamp
- Actual demand
- Forecasted demand
- Forecast error

Output file:

`outputs/predictions/hourly_demand_forecast.csv`

The final forecast visualization contains:

- Historical training demand
- Actual test-period demand
- Prophet forecast

Output figure:

`outputs/figures/hourly_demand_forecast.png`

---

## 13. Business Interpretation

The forecasting analysis identified several operationally relevant patterns.

### 13.1 Evening Demand Peak

The strongest demand occurs between **18:00 and 22:00**, with 22:00 having the highest average demand at **21.62 orders/hour**.

This suggests that operational capacity should be increased during the evening peak.

Potential actions include:

- Increasing rider availability
- Preparing restaurants for higher order volumes
- Increasing operational monitoring
- Preparing dispatch capacity for peak periods

---

### 13.2 Higher Weekend Demand

Weekend demand averaged **15.04 orders/hour**, compared with **10.09 orders/hour** on weekdays.

This represents approximately **49% higher demand on weekends**.

Therefore, capacity planning should distinguish between weekday and weekend demand rather than using a single average capacity level.

---

### 13.3 Time-Based Resource Allocation

Demand varies significantly throughout the day.

The lower-demand periods and high-demand periods suggest that resources can be allocated dynamically rather than uniformly across all hours.

Forecast information can support:

- Rider scheduling
- Restaurant staffing
- Dispatch planning
- Operational monitoring
- Capacity management

---

### 13.4 Proactive Operational Planning

Forecasting provides an opportunity to prepare for expected demand before it occurs.

Higher forecasted demand can trigger additional operational capacity, while lower forecasted demand can help avoid unnecessary over-allocation of resources.

This makes demand forecasting useful for proactive marketplace operations.

---

## 14. Limitations

The results should be interpreted with caution because the project uses a **synthetic dataset**.

Important limitations include:

- The demand patterns were artificially generated.
- Seasonal patterns may be more predictable than those in real-world food delivery data.
- Forecasting performance may therefore be more favorable than production performance.
- The dataset contains only six months of historical data.
- The forecasting model does not incorporate external demand drivers such as promotions, holidays, major events, pricing changes, or restaurant availability.
- MAPE can be sensitive to low-demand observations.

The reported model performance should therefore be viewed as a demonstration of the forecasting methodology rather than real-world Uber Eats forecasting accuracy.

---

## 15. Future Improvements

A production-oriented forecasting system could be improved by:

- Using longer historical data
- Incorporating holidays and special events
- Including promotional activity
- Including restaurant and rider availability
- Incorporating weather information
- Modeling city-level demand separately
- Evaluating seasonal ARIMA/SARIMA approaches
- Performing rolling or walk-forward validation
- Tuning Prophet hyperparameters
- Comparing additional forecasting algorithms
- Monitoring forecast accuracy over time

---

## 16. Key Takeaways

The hourly demand analysis identified strong **daily and weekly seasonality** in the synthetic Uber Eats order data.

The most important findings were:

1. **18:00–22:00 is the primary demand peak.**
2. **22:00 has the highest average demand at 21.62 orders/hour.**
3. **Weekend demand is approximately 49% higher than weekday demand.**
4. **Prophet outperformed the simple ARIMA baseline across MAE, RMSE, and MAPE.**
5. **Prophet was selected as the final forecasting model.**
6. Forecasting can support proactive **rider, restaurant, and operational capacity planning**.
7. The results should be interpreted as a demonstration using synthetic data rather than production-level Uber Eats performance.