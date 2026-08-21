# Uber Eats Operational & Customer Intelligence System

## 1. Project Overview

The **Uber Eats Operational & Customer Intelligence System** is an end-to-end data analytics and machine learning project built around a synthetic food-delivery marketplace dataset.

The project combines customer analytics, natural language processing, delivery-time prediction, and demand forecasting to understand customer behavior, identify operational issues, predict delivery performance, and support business decision-making.

The project follows a complete analytical workflow:

**Data Generation → Data Validation → Feature Engineering → **EDA** → Machine Learning → Model Evaluation → Business Insights → Recommendations**

> **Note:** This project uses synthetic Uber Eats-style data created for analytical and machine-learning practice. The results demonstrate the analytical workflow and business interpretation and should not be treated as real Uber Eats production performance.

---

## 2. Objectives

The main objectives of the project are to:

- Segment customers based on their ordering behavior and value.
- Identify high-value customer groups.
- Analyze customer reviews using Natural Language Processing (**NLP**).
- Classify customer sentiment.
- Identify common customer complaint themes.
- Predict delivery time using machine learning.
- Analyze the operational factors affecting delivery time.
- Forecast hourly order demand.
- Identify peak-demand periods.
- Translate machine-learning outputs into actionable business recommendations.
- Build a structured and reproducible analytics pipeline using Python.

---

## 3. Project Modules

The project consists of the following major analytical modules.

### 3.1 Customer Segmentation

Customer-level behavioral features are created from completed orders and used to identify customer segments.

Key features include:

- Total orders
- Average order value
- Total spending
- Ordering frequency
- Average rating given
- Weekend orders
- Late-night orders

The segmentation workflow includes:

- Feature engineering
- Exploratory Data Analysis
- Feature scaling
- K-Means clustering
- Elbow method
- Silhouette analysis
- Cluster profiling
- Business interpretation

The analysis identified two major customer segments:

- **Highly Engaged / High-Value Customers**
- **Occasional / Lower-Engagement Customers**

---

### 3.2 Customer Feedback Analysis Using NLP

Customer reviews are analyzed using Natural Language Processing techniques.

The **NLP** pipeline includes:

- Text preprocessing
- Tokenization
- Stop-word removal
- Negation preservation
- Lemmatization
- TF-**IDF** feature extraction
- Train/test splitting
- Linear **SVM** sentiment classification
- Model evaluation
- **VADER** sentiment analysis
- Negative-review theme analysis

The main complaint categories identified include:

- Delivery Delays
- Food Quality / Temperature
- Wrong / Missing Items
- Packaging Damage

---

### 3.3 Delivery-Time Prediction

A machine-learning model is used to predict delivery time.

The delivery prediction workflow includes:

- Feature engineering
- Train/test splitting
- Model training
- Prediction
- Error analysis
- Actual vs predicted analysis
- Feature importance analysis
- Business interpretation

Delivery performance is also analyzed across:

- Traffic conditions
- Weather
- Delivery distance
- Restaurant preparation time

---

### 3.4 Demand Forecasting

Hourly order demand is forecasted to support operational capacity planning.

The demand forecasting workflow includes:

- Time-series preparation
- Hourly demand aggregation
- Forecast generation
- Forecast evaluation
- Peak-hour analysis
- Weekend vs weekday analysis
- Forecast bias analysis

The forecast is used to identify periods where additional operational capacity may be required.

---

## 4. Dataset

The project uses synthetic Uber Eats-style marketplace data.

The generated datasets include:

| Dataset | Description |
|---|---|
| `customers.csv` | Customer information |
| `restaurants.csv` | Restaurant information |
| `drivers.csv` | Driver information |
| `orders.csv` | Order-level transaction data |
| `deliveries.csv` | Delivery and operational information |
| `reviews.csv` | Customer review and sentiment data |
| `payments.csv` | Payment information |

### Dataset Size

The synthetic data contains approximately:

- **5,**000** customers**
- ****300** restaurants**
- ****500** drivers**
- **50,**000** orders**
- **50,**000** deliveries**
- **30,**000** reviews**
- **50,**000** payments**

A fixed random seed was used during data generation to make the dataset reproducible.

### Data Validation

The generated datasets were validated for:

- Duplicate IDs
- Referential integrity
- Missing relationships
- Order and delivery consistency
- Review validity
- Business logic consistency

The validation checks confirmed that the generated datasets maintained the expected relationships between entities.

---

## 5. Technology Stack

### Programming Language

- Python

### Data Analysis

- Pandas
- NumPy

### Machine Learning

- Scikit-learn

### Natural Language Processing

- **NLTK**
- TF-**IDF**
- Linear **SVM**
- **VADER**

### Statistical / Forecasting Analysis

- Statsmodels
- Prophet

### Visualization

- Matplotlib
- Seaborn

### Development Tools

- VS Code
- Git
- GitHub
- Python virtual environment

---

## 6. Project Structure

Uber_Eats_Operational_Customer_Intelligence/
│
├── data/
│   ├── raw/
│   │   ├── customers.csv
│   │   ├── restaurants.csv
│   │   ├── drivers.csv
│   │   ├── orders.csv
│   │   ├── deliveries.csv
│   │   ├── reviews.csv
│   │   └── payments.csv
│   │
│   └── processed/
│       ├── customer_features.csv
│       ├── customer_segments.csv
│       ├── reviews_cleaned.csv
│       ├── tfidf_features.csv
│       └── negative_review_themes.csv
│
├── docs/
│   ├── data_dictionary.md
│   ├── data_generation_notes.md
│   ├── feature_engineering.md
│   ├── segmentation.md
│   └── customer_feedback_analysis.md
│
├── outputs/
│   ├── figures/
│   │   └── sentiment_confusion_matrix.png
│   │
│   ├── models/
│   │
│   └── predictions/
│       ├── sentiment_predictions.csv
│       ├── delivery_time_predictions.csv
│       └── hourly_demand_forecast.csv
│
├── src/
│   ├── __init__.py
│   ├── business_insights.py
│   ├── data_cleaning.py
│   ├── data_generation.py
│   ├── delivery_prediction.py
│   ├── demand_forecasting.py
│   ├── feature_engineering.py
│   ├── nlp_sentiment.py
│   └── segmentation.py
│
├── presentation/
│
├── main.py
├── README.md
└── requirements.txt

---

## 7. Machine Learning Workflow

The project follows a modular machine-learning workflow.

### Step 1: Data Generation

Synthetic marketplace data is generated for customers, restaurants, drivers, orders, deliveries, reviews, and payments.

### Step 2: Data Validation

The generated datasets are checked for:

- Missing values
- Duplicate records
- Referential integrity
- Invalid relationships
- Business logic consistency

### Step 3: Feature Engineering

Raw transactional data is transformed into analytical features required for machine-learning models and business analysis.

Examples include:

- Customer spending features
- Customer ordering behavior
- Delivery distance
- Restaurant preparation time
- Traffic and weather variables
- Time-based demand features

### Step 4: Exploratory Data Analysis

**EDA** is performed to understand:

- Feature distributions
- Relationships between variables
- Customer behavior
- Delivery-time patterns
- Demand patterns
- Customer feedback

### Step 5: Customer Segmentation

Customer features are standardized and analyzed using clustering techniques.

K-Means clustering is used to identify behavioral customer segments.

### Step 6: NLP Sentiment Classification

Customer reviews are transformed using TF-**IDF** and classified using a Linear **SVM** model.

**VADER** is also used as a rule-based sentiment analysis approach for comparison.

### Step 7: Delivery-Time Prediction

Machine-learning models are used to predict delivery duration and evaluate prediction errors.

### Step 8: Demand Forecasting

Historical hourly demand is used to generate demand forecasts and evaluate forecasting performance.

### Step 9: Business Interpretation

Model outputs are converted into:

- Customer insights
- Operational insights
- Customer-experience insights
- Capacity-planning recommendations
- Business recommendations

---

## 8. Key Results

### Customer Segmentation

The analysis identified two customer segments.

| SegmentCustomersAvg. Total OrdersAvg. Order ValueAvg. Total Spending |       |       |        |          |
| -------------------------------------------------------------------- | ----- | ----- | ------ | -------- |
| Highly Engaged / High-Value Customers                                | 2,181 | 11.80 | 591.85 | 6,921.34 |
| Occasional / Lower-Engagement Customers                              | 2,819 | 7.21  | 549.17 | 3,909.19 |

The **Highly Engaged / High-Value Customers** segment has the highest average spending at **6,**921**.34 per customer**.

---

### Customer Sentiment

The evaluated reviews were distributed as follows:

| SentimentReviewsPercentage |       |        |
| -------------------------- | ----- | ------ |
| Positive                   | 4,172 | 69.53% |
| Negative                   | 1,140 | 19.00% |
| Neutral                    | 688   | 11.47% |

The most common negative-review theme was **Delivery Delays**, representing approximately **40.09%** of categorized negative reviews.

Other major complaint themes included:

- Food Quality / Temperature — 30.11%
- Wrong / Missing Items — 19.19%
- Packaging Damage — 9.67%

---

### Delivery-Time Prediction

The delivery-time prediction model achieved:

- ****MAE**: 3.69 minutes**
- **Mean Prediction Error: 0.03 minutes**

Observed average delivery times increased with several operational factors.

#### Traffic

- Low traffic: **44.69 minutes**
- Medium traffic: **55.46 minutes**
- High traffic: **66.65 minutes**

#### Weather

- Clear: **53.37 minutes**
- Cloudy: **55.43 minutes**
- Rain: **61.10 minutes**
- Storm: **68.44 minutes**

#### Distance

- 0–2 km: **44.43 minutes**
- 2–5 km: **52.17 minutes**
- 5–10 km: **63.86 minutes**
- 10+ km: **84.55 minutes**

#### Preparation Time

- 0–15 minutes: **44.42 minutes**
- 15–30 minutes: **54.14 minutes**
- 30–45 minutes: **67.97 minutes**
- 45+ minutes: **80.25 minutes**

Among the analyzed factors, **delivery distance** showed the largest observed difference in average delivery time at approximately **40.12 minutes**.

---

### Demand Forecasting

The demand forecasting model achieved:

- ****MAE**: 3.11 orders/hour**
- ****RMSE**: 4.10 orders/hour**
- ****MAPE**: 38.86%**

Average demand during the forecast period was:

- Actual: **11.56 orders/hour**
- Forecasted: **11.65 orders/hour**

The highest average demand occurred at approximately **21:00**, with average demand of **22.97 orders/hour**.

Weekend demand was also higher than weekday demand:

- Weekday: **10.17 orders/hour**
- Weekend: **15.19 orders/hour**

---

## 9. Business Insights

### Customer Value

Highly Engaged / High-Value Customers spend significantly more than Occasional / Lower-Engagement Customers.

The difference in average spending is approximately **3,**012**.15 per customer**.

This segment therefore represents an important customer group for retention and loyalty strategies.

---

### Customer Experience

Approximately **19%** of evaluated reviews are negative.

**Delivery Delays** are the most common categorized complaint, accounting for approximately **40.09%** of negative reviews.

This indicates that delivery performance is an important area for improving customer experience.

---

### Delivery Operations

Delivery distance has the largest observed effect on average delivery time among the analyzed factors.

Restaurant preparation time is the second-largest observed factor.

Traffic conditions also show a substantial difference:

- Low traffic: 44.69 minutes
- High traffic: 66.65 minutes

This indicates that operational planning should account for both restaurant-side and delivery-side constraints.

---

### Demand and Capacity

Demand is highest during evening hours, particularly around **21:00**.

Weekend demand is substantially higher than weekday demand.

This suggests that rider availability and operational capacity should be adjusted during:

- Evening peak periods
- Weekend periods
- Other high-demand intervals identified through forecasting

---

### Forecasting

The demand forecasting model provides a useful baseline for capacity planning.

However, individual demand spikes can still produce larger forecasting errors, so forecasts should be monitored against actual demand.

---

## 10. Business Recommendations

### 1. Retain High-Value Customers

Prioritize the **Highly Engaged / High-Value Customers** segment through:

- Loyalty rewards
- Personalized offers
- Targeted promotions
- Retention strategies

---

### 2. Reduce Delivery Delays

Since Delivery Delays are the largest categorized complaint theme:

- Monitor long-distance orders.
- Identify restaurants with consistently long preparation times.
- Use traffic information when planning deliveries.
- Improve estimated delivery-time calculations.
- Investigate operational causes of repeated delays.

---

### 3. Plan Capacity Around Peak Demand

Increase operational capacity during high-demand periods.

Potential actions include:

- Increasing rider availability
- Adjusting staffing levels
- Monitoring restaurant capacity
- Preparing for evening peaks
- Allocating additional resources on weekends

---

### 4. Improve Customer Experience

Use customer feedback to prioritize operational improvements.

Since Delivery Delays represent the largest complaint category, reducing delivery-time variability and improving **ETA** accuracy can potentially improve customer satisfaction.

---

### 5. Use Forecasts for Operational Planning

Demand forecasts can support:

- Rider allocation
- Staffing decisions
- Restaurant capacity planning
- Peak-period preparation
- Resource allocation

Forecasts should be combined with real-time monitoring because unexpected demand spikes may not be predicted accurately.

---

## 13. How to Run the Project

### 1. Clone the Repository

``` git clone <repository-url> cd Uber_Eats_Operational_Customer_Intelligence ```

### 2. Create a Virtual Environment

``` python -m venv .venv ```

### 3. Activate the Virtual Environment

#### Linux / macOS

``` source .venv/bin/activate ```

#### Windows

``` .venv\Scripts\activate ```

### 4. Install Dependencies

``` pip install -r requirements.txt ```

### 5. Run the Project

The project modules can be executed from the `src/` directory according to the analytical workflow.

For example:

``` python src/data_generation.py```

``` python src/feature_engineering.py ```

``` python src/segmentation.py ```

``` python src/nlp_sentiment.py ```

``` python src/delivery_prediction.py ```

``` python src/demand_forecasting.py ```

``` python src/business_insights.py ```

The generated datasets are stored under `data/`, while processed datasets, model outputs, predictions, and visualizations are stored under `data/processed/` and `outputs/`.

---