# Delivery Time Prediction

## 1. Overview

Delivery time prediction is a supervised machine learning task designed to estimate the time required to complete an Uber Eats delivery.

The objective of this module is to predict delivery time using operational, order, restaurant, traffic, weather, and time-related features.

The workflow includes:

- Data preparation
- Filtering successfully delivered orders
- Dataset merging
- Feature engineering
- Exploratory data analysis
- Train/test splitting
- Numerical and categorical preprocessing
- Regression modeling
- Model comparison
- Cross-validation
- Feature importance analysis
- Prediction generation
- Business interpretation

The final selected model is a **Linear Regression pipeline**, which achieved the strongest overall performance among the evaluated models.

---

## 2. Objective

The primary objective is to predict:

**`delivery_time_min`**

using information available about the order, restaurant, delivery conditions, and time of ordering.

The prediction can support:

- More realistic delivery time estimates
- Operational planning
- Rider allocation
- Delay monitoring
- Restaurant preparation monitoring
- Traffic- and weather-aware ETA adjustments

---

## 3. Data Sources

The delivery prediction module uses three raw datasets:

- `orders.csv`
- `deliveries.csv`
- `restaurants.csv`

The datasets are stored under:

`data/raw/`

### Delivery Data

The delivery dataset provides information such as:

- Delivery distance
- Preparation time
- Delivery time
- Delivery status
- Driver information

### Order Data

The order dataset provides:

- Order timestamp
- Order amount
- Number of items
- Order status
- Customer and restaurant information
- Weather
- Traffic condition

### Restaurant Data

The restaurant dataset provides:

- Restaurant name
- Restaurant category
- Restaurant rating
- Average preparation time
- Restaurant information

---

## 4. Data Preparation

Only successfully delivered orders were used for modeling.

Records with:

`delivery_status != "Delivered"`

were excluded because the target variable represents the actual delivery time of completed deliveries.

### Dataset Filtering

The original delivery dataset contained:

- **50,000 delivery records**

After filtering for successfully delivered orders:

- **46,046 delivered records**

Therefore:

- **3,954 failed/non-delivered records were removed**

The resulting delivery-time target contained no zero-valued delivery times.

---

## 5. Dataset Integration

The delivery data was merged with order information using:

`order_id`

Restaurant information was then merged using:

`restaurant_id`

The resulting modeling dataset contained:

- **46,046 rows**
- **22 columns**

Data validation showed:

- No missing values
- No duplicate rows
- No zero delivery times after filtering

This provided a clean dataset for regression modeling.

---

## 6. Feature Engineering

Time-based features were created from the order timestamp.

### Hour

The hour of the day was extracted from:

`order_timestamp`

This captures differences in delivery behavior throughout the day.

### Day of Week

The day of the week was extracted as an integer from 0 to 6.

This allows the model to capture differences between weekdays and weekends.

### Weekend Indicator

A binary feature was created:

- `1` → Saturday or Sunday
- `0` → Monday to Friday

This feature captures potential weekend-related delivery patterns.

---

## 7. Modeling Features

The final model used the following features:

### Numerical Features

- `delivery_distance_km`
- `preparation_time_min`
- `order_amount`
- `num_items`
- `restaurant_rating`
- `avg_prep_time`
- `hour`
- `day_of_week`
- `is_weekend`

### Categorical Features

- `weather`
- `traffic_condition`
- `restaurant_category`

### Target Variable

The prediction target was:

`delivery_time_min`

---

## 8. Exploratory Data Analysis

Before modeling, the distribution and relationships of the target variable were analyzed.

### Delivery Time Distribution

The target variable had the following statistics:

| Statistic | Delivery Time |
|---|---:|
| Minimum | 10.0 min |
| 25th Percentile | 43.3 min |
| Median | 54.8 min |
| 75th Percentile | 66.7 min |
| 95th Percentile | 86.0 min |
| 99th Percentile | 101.0 min |
| Maximum | 143.3 min |

The mean delivery time was approximately:

**55.62 minutes**

The target skewness was:

**0.358**

This indicates a moderate right-skewed distribution rather than an extremely skewed target.

---

## 9. Correlation Analysis

The relationship between selected numerical features and delivery time was examined.

### Delivery Distance

Correlation with delivery time:

**0.591**

This was the strongest correlation among the numerical features examined.

### Preparation Time

Correlation with delivery time:

**0.543**

Longer restaurant preparation times were associated with longer delivery times.

### Average Preparation Time

Correlation with delivery time:

**0.512**

Restaurants with higher average preparation times tended to have longer delivery times.

These relationships indicate that both delivery distance and restaurant preparation are important operational drivers.

---

## 10. Delivery Time by Traffic Condition

Average delivery time varied substantially by traffic condition.

| Traffic Condition | Mean Delivery Time |
|---|---:|
| Low | 44.69 min |
| Medium | 55.46 min |
| High | 66.65 min |

High-traffic deliveries took approximately **22 minutes longer on average** than low-traffic deliveries.

This indicates that traffic conditions are an important factor for delivery-time prediction and ETA planning.

---

## 11. Delivery Time by Weather

Weather conditions also showed clear differences.

| Weather | Mean Delivery Time |
|---|---:|
| Clear | 53.37 min |
| Cloudy | 55.43 min |
| Rain | 61.10 min |
| Storm | 68.44 min |

Storm conditions had the highest average delivery time.

This indicates that adverse weather conditions can contribute to longer delivery times.

---

## 12. Delivery Time by Distance

Delivery time increased as delivery distance increased.

| Distance | Mean Delivery Time |
|---|---:|
| 0–2 km | 44.43 min |
| 2–5 km | 52.17 min |
| 5–10 km | 63.86 min |
| 10+ km | 84.55 min |

Deliveries exceeding 10 km had substantially higher delivery times than short-distance deliveries.

This demonstrates the importance of delivery distance as a predictive and operational feature.

---

## 13. Delivery Time by Preparation Time

Restaurant preparation time also showed a strong relationship with total delivery time.

| Preparation Time | Mean Delivery Time |
|---|---:|
| 0–15 min | 44.42 min |
| 15–30 min | 54.14 min |
| 30–45 min | 67.97 min |
| 45+ min | 80.25 min |

Orders requiring longer preparation times experienced substantially longer overall delivery times.

This suggests that restaurant preparation efficiency is an important component of delivery performance.

---

## 14. Train/Test Split

The dataset was divided into training and testing subsets using an 80/20 split.

| Dataset | Records |
|---|---:|
| Training | 36,836 |
| Testing | 9,210 |

A fixed random state of 42 was used to ensure reproducibility.

---

## 15. Data Preprocessing

A `ColumnTransformer` was used to apply different preprocessing strategies to numerical and categorical features.

### Numerical Features

Numerical features were standardized using:

**StandardScaler**

Standardization ensures that numerical variables are placed on comparable scales, which is particularly useful for linear models and regularized regression.

### Categorical Features

Categorical variables were transformed using:

**OneHotEncoder**

The encoder used:

`handle_unknown="ignore"`

This prevents errors when unseen categories appear during prediction.

---

## 16. Regression Models

Four regression approaches were evaluated.

### 16.1 Linear Regression

Linear Regression was used as the baseline model.

The complete preprocessing and regression process was implemented as a single sklearn pipeline.

### 16.2 Random Forest Regressor

Random Forest was evaluated as a nonlinear ensemble model.

The model used:

- 100 trees
- Random state = 42
- Parallel processing using all available CPU cores

### 16.3 Ridge Regression

Ridge Regression was evaluated to determine whether L2 regularization improved the baseline model.

The model used:

`alpha = 1.0`

### 16.4 Lasso Regression

Lasso Regression was evaluated to determine whether L1 regularization could improve performance or provide useful feature selection.

The model used:

`alpha = 1.0`

---

## 17. Model Evaluation

Models were evaluated using:

### Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual and predicted delivery time.

Lower values indicate better performance.

### Root Mean Squared Error (RMSE)

RMSE penalizes larger prediction errors more strongly than MAE.

Lower values indicate better performance.

### R² Score

R² measures the proportion of variance in delivery time explained by the model.

Higher values indicate better performance.

---

## 18. Test Set Model Comparison

The models were evaluated on the held-out test set.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| **Linear Regression** | **3.6921** | **4.6233** | **0.9315** |
| Ridge Regression | 3.6921 | 4.6233 | 0.9315 |
| Random Forest | 3.7816 | 4.7517 | 0.9276 |
| Lasso Regression | 5.1301 | 6.4663 | 0.8660 |

Linear Regression produced the strongest overall test-set performance.

---

## 19. Five-Fold Cross-Validation

Five-fold cross-validation was performed to assess model stability and reduce dependence on a single train/test split.

The data was shuffled before creating the folds using a fixed random state of 42.

### Cross-Validation Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| **Linear Regression** | **3.7146 ± 0.0179** | **4.6628 ± 0.0280** | **0.9291 ± 0.0013** |
| Ridge Regression | 3.7146 ± 0.0179 | 4.6628 ± 0.0280 | 0.9291 ± 0.0013 |
| Random Forest | 3.7983 ± 0.0261 | 4.7757 ± 0.0296 | 0.9256 ± 0.0012 |
| Lasso Regression | 5.0865 ± 0.0492 | 6.4321 ± 0.0498 | 0.8650 ± 0.0013 |

The small standard deviations across folds indicate stable model performance on the synthetic dataset.

---

## 20. Final Model Selection

**Linear Regression** was selected as the final model.

The final model is stored in the Python variable:

`baseline_model`

### Reasons for Selection

1. It achieved the lowest cross-validation MAE.
2. It achieved the lowest cross-validation RMSE.
3. It achieved the highest cross-validation R².
4. Ridge Regression provided no measurable improvement.
5. Random Forest did not outperform the simpler linear model.
6. Lasso Regression produced substantially weaker results.

The selected model therefore provides the best combination of predictive performance and simplicity among the tested approaches.

---

## 21. Feature Importance

For Linear Regression, feature coefficients were examined after preprocessing.

The strongest coefficients included:

| Feature | Coefficient |
|---|---:|
| Traffic – High | 10.44 |
| Delivery Distance | 10.32 |
| Traffic – Low | -10.08 |
| Preparation Time | 9.59 |
| Weather – Storm | 8.77 |
| Weather – Clear | -6.22 |
| Weather – Cloudy | -4.26 |
| Weather – Rain | 1.71 |

The coefficient analysis indicates that traffic, delivery distance, preparation time, and weather conditions are important contributors to predicted delivery time.

For categorical variables, the coefficients represent effects relative to the reference category created by the one-hot encoding process.

---

## 22. Actual vs Predicted Analysis

An actual-versus-predicted visualization was created to evaluate how closely model predictions followed the observed delivery times.

The plot compares:

- Actual delivery time
- Predicted delivery time

A 45-degree reference line represents perfect prediction.

The visualization is saved as:

`outputs/figures/actual_vs_predicted_delivery_time.png`

---

## 23. Prediction Output

Predictions were generated for the 9,210 test-set records.

The prediction output contains:

- Modeling features
- Actual delivery time
- Predicted delivery time

The output is saved to:

`outputs/predictions/delivery_time_predictions.csv`

Example output fields include:

- `delivery_distance_km`
- `preparation_time_min`
- `order_amount`
- `num_items`
- `weather`
- `traffic_condition`
- `restaurant_category`
- `restaurant_rating`
- `avg_prep_time`
- `hour`
- `day_of_week`
- `is_weekend`
- `actual_delivery_time`
- `predicted_delivery_time`

---

## 24. Business Interpretation

The analysis identified several operational factors associated with longer delivery times.

### Traffic

High-traffic deliveries averaged:

**66.65 minutes**

compared with:

**44.69 minutes**

for low-traffic deliveries.

Traffic should therefore be incorporated into ETA estimation and delivery operations.

### Weather

Storm conditions had an average delivery time of:

**68.44 minutes**

compared with:

**53.37 minutes**

under clear conditions.

Adverse weather should therefore be considered when estimating delivery times.

### Distance

Deliveries above 10 km averaged:

**84.55 minutes**

compared with:

**44.43 minutes**

for deliveries between 0 and 2 km.

Long-distance deliveries require additional ETA and operational consideration.

### Restaurant Preparation

Deliveries with preparation times above 45 minutes averaged:

**80.25 minutes**

compared with:

**44.42 minutes**

for preparation times below 15 minutes.

Restaurant preparation efficiency is therefore an important operational factor.

---

## 25. Business Recommendations

### 1. Traffic-Aware Delivery Planning

High traffic is associated with substantially longer delivery times.

Uber Eats can incorporate real-time traffic conditions into ETA estimation and rider allocation decisions.

### 2. Weather-Aware ETA Adjustment

Rain and storm conditions are associated with longer delivery times.

ETA systems should account for adverse weather conditions when generating delivery estimates.

### 3. Distance-Based Delivery Planning

Long-distance deliveries have significantly higher expected delivery times.

Orders exceeding certain distance thresholds can receive appropriate ETA adjustments and additional operational monitoring.

### 4. Restaurant Preparation Monitoring

Restaurants with consistently high preparation times can be identified for operational improvement.

Reducing preparation delays can contribute to lower overall delivery times.

### 5. Predictive ETA Usage

The final model achieved a test-set MAE of approximately:

**3.69 minutes**

The mean prediction error was approximately:

**0.03 minutes**

This indicates very little systematic prediction bias on the test set.

The model can therefore support more realistic delivery-time estimates within the limitations of the synthetic dataset.

---

## 26. Model Limitations

The model has several limitations.

### Synthetic Dataset

The dataset used in this project is synthetically generated.

Therefore, the reported performance should not be interpreted as production-level performance on real Uber Eats data.

### Feature Availability

The model assumes that variables such as traffic, weather, restaurant preparation time, and delivery distance are available at prediction time.

In a real production system, feature availability and latency would need to be considered.

### Random Train/Test Split

The model uses a random train/test split.

For a real-world deployment scenario, a time-based validation strategy could be more appropriate because delivery patterns may change over time.

### Model Complexity

Only a limited set of regression algorithms was evaluated.

More advanced models such as Gradient Boosting, XGBoost, LightGBM, or other ensemble approaches could be evaluated in future work.

### Synthetic Relationships

The strong model performance may partly reflect predictable relationships embedded in the synthetic data generation process.

Therefore, the results should primarily be interpreted as a demonstration of the machine learning workflow.

---

## 27. Future Improvements

Potential improvements include:

- Use real-world delivery data
- Use time-based validation
- Perform hyperparameter tuning
- Evaluate Gradient Boosting, XGBoost, or LightGBM
- Add geographic and driver-history features
- Incorporate real-time traffic and weather data
- Implement model monitoring and drift detection

---

## 28. Final Conclusion

The delivery-time prediction module successfully demonstrates an end-to-end supervised regression workflow.

The process included:

**Raw Data → Filtering → Data Integration → Feature Engineering → EDA → Preprocessing → Regression → Cross-Validation → Model Selection → Prediction → Business Interpretation**

Four regression approaches were evaluated:

- Linear Regression
- Ridge Regression
- Random Forest
- Lasso Regression

Linear Regression was selected as the final model because it achieved the best overall performance across both the held-out test set and five-fold cross-validation, while remaining simpler and more interpretable than Random Forest.

The final model achieved:

- **Test MAE:** 3.69 minutes
- **Test RMSE:** 4.62 minutes
- **Test R²:** 0.9315
- **5-Fold CV MAE:** 3.71 minutes
- **5-Fold CV RMSE:** 4.66 minutes
- **5-Fold CV R²:** 0.9291

The analysis also identified traffic, delivery distance, preparation time, and weather as important operational factors associated with delivery time.

The prediction output was saved to `outputs/predictions/delivery_time_predictions.csv` for analysis and downstream business interpretation.

Overall, the module demonstrates how supervised machine learning can be used to predict delivery times and translate model results into operational decisions for a food-delivery marketplace.