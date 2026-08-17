# Synthetic Data Generation

## Objective

Generate realistic synthetic datasets representing an Uber Eats-style
food delivery marketplace for machine learning and business analysis.

## Customer Dataset

The first synthetic dataset represents customers.

### Dataset Size

- Records: 5,000
- Features: 5

### Features

| Feature | Description |
|---|---|
| customer_id | Unique customer identifier |
| age | Customer age between 18 and 65 |
| gender | Customer gender |
| city | Customer location |
| signup_date | Customer registration date |

### Validation

The generated customer dataset was validated for:

- Dataset shape
- Duplicate customer IDs
- Missing values
- Age range
- Valid gender categories

### Validation Results

- Shape: 5,000 × 5
- Duplicate customer IDs: 0
- Missing values: 0
- Age range: 18–65
- Gender categories: Male, Female

The dataset was saved to:

`data/raw/customers.csv`


## Restaurant Dataset

The second synthetic dataset represents restaurants available on the food
delivery platform.

### Dataset Size

- Records: 300
- Features: 6

### Features

| Feature | Description |
|---|---|
| restaurant_id | Unique restaurant identifier |
| restaurant_name | Synthetic restaurant name |
| restaurant_category | Type of cuisine/restaurant |
| restaurant_rating | Restaurant rating between 3.0 and 5.0 |
| avg_prep_time | Average food preparation time in minutes |
| city | Restaurant location |

### Restaurant Categories

The dataset contains the following categories:

- Indian
- Chinese
- Fast Food
- Biryani
- Pizza
- South Indian
- Desserts
- Cafe
- Bakery
- Healthy

### Generation Logic

Restaurant preparation time was generated based on restaurant category rather
than being completely random.

For example:

- Fast Food, Bakery and Desserts generally have shorter preparation times.
- Pizza and Chinese restaurants have moderate preparation times.
- Indian and Biryani restaurants generally have longer preparation times.

Random variation was introduced within each category to make the synthetic
data more realistic.

Restaurant ratings were generated within a constrained range of 3.0 to 5.0, with values concentrated around 4.1.

### Validation

The generated restaurant dataset was validated for:

- Dataset shape
- Duplicate restaurant IDs
- Missing values
- Restaurant rating range
- Preparation time range
- Category distribution

### Validation Results

- Shape: 300 × 6
- Duplicate restaurant IDs: 0
- Missing values: 0
- Rating range: 3.0–5.0
- Preparation time range: 10–50 minutes
- All 10 restaurant categories were represented.

### Relationship Validation

The relationship between restaurant category and average preparation time was
also checked.

Biryani restaurants showed the highest average preparation time, while
Bakery, Desserts and Fast Food showed lower average preparation times.

This relationship was intentionally introduced so that restaurant category can
contribute to downstream operational analysis and delivery-time prediction.

The dataset was saved to:

`data/raw/restaurants.csv`



## Driver Dataset

The third synthetic dataset represents delivery drivers operating on the
food delivery platform.

### Dataset Size

- Records: 500
- Features: 5

### Features

| Feature | Description |
|---|---|
| driver_id | Unique driver identifier |
| vehicle_type | Driver's delivery vehicle |
| driver_rating | Driver rating between 3.0 and 5.0 |
| experience_years | Delivery experience in years |
| city | Driver's operating city |

### Vehicle Types

The dataset contains three vehicle types:

- Bike
- Scooter
- Car

Bikes were intentionally made the most common vehicle type, followed by
scooters and cars, reflecting the suitability of two-wheelers for food
delivery operations.

### Generation Logic

Driver ratings were generated within a constrained range of approximately
3.0 to 5.0, with most drivers concentrated around higher ratings.

Experience was generated between 0 and 10 years, with a higher proportion of
drivers having lower to moderate experience.

Drivers were assigned to the same set of cities used for customers and
restaurants so that location-based relationships can be maintained when
generating orders and deliveries.

### Validation

The generated driver dataset was validated for:

- Dataset shape
- Duplicate driver IDs
- Missing values
- Driver rating range
- Experience range
- Vehicle distribution

### Validation Results

- Shape: 500 × 5
- Duplicate driver IDs: 0
- Missing values: 0
- Rating range: 3.2–5.0
- Experience range: 0–10 years
- Vehicle types: Bike, Scooter, Car

The dataset was saved to:

`data/raw/drivers.csv`



## Orders Dataset

The Orders dataset acts as the central transaction table of the synthetic
Uber Eats marketplace. It connects customers, restaurants and drivers and
contains the operational information required for downstream ML tasks.

### Dataset Size

- Records: 50,000
- Features: 11
- Time period: January 1, 2026 to June 30, 2026

### Features

| Feature | Description |
|---|---|
| order_id | Unique order identifier |
| customer_id | Customer associated with the order |
| restaurant_id | Restaurant associated with the order |
| driver_id | Driver assigned to the order |
| order_timestamp | Timestamp when the order was placed |
| order_amount | Total food/order value |
| num_items | Number of items in the order |
| order_status | Completed or Cancelled |
| city | City where the order occurred |
| weather | Weather condition |
| traffic_condition | Traffic condition |

### Relationships

Orders were generated while maintaining referential integrity with the
Customers, Restaurants and Drivers datasets.

- Each customer ID exists in the customer dataset.
- Each restaurant ID exists in the restaurant dataset.
- Each driver ID exists in the driver dataset.
- Orders are associated with restaurants and drivers operating
  within the customer's city.

### Temporal Demand Pattern

Order timestamps were generated over a six-month period.

The synthetic data intentionally includes realistic demand patterns:

- Higher demand during lunch hours.
- Stronger demand during dinner hours.
- Lower demand during early morning and non-peak periods.
- Increased weekend demand.

This creates meaningful temporal patterns for the demand forecasting task.

### Order Amount

Order amount was generated based on the number of items and a randomly
generated base item price.

Therefore, larger orders generally have higher order values.

The generated order amount ranged from approximately ₹80 to ₹2,100.

### Traffic

Traffic conditions were generated based partly on the order time.

Peak lunch and dinner periods have a higher probability of Medium or High
traffic, while non-peak periods have a higher probability of Low traffic.

This relationship will be useful for downstream delivery-time prediction.

### Weather

Four weather conditions were generated:

- Clear
- Cloudy
- Rain
- Storm

Clear weather was intentionally made the most common condition.

### Order Status

Orders were generated with two statuses:

- Completed
- Cancelled

Approximately 92% of orders were completed and 8% were cancelled.

### Validation

The dataset was validated for:

- Dataset shape
- Duplicate order IDs
- Missing values
- Order amount range
- Number of items
- Date range
- Order status distribution
- Weather distribution
- Traffic distribution
- Referential integrity

### Validation Results

- Shape: 50,000 × 11
- Duplicate order IDs: 0
- Missing values: 0
- Order amount range: ₹80–₹2,099.80
- Number of items: 1–6
- Invalid customer IDs: 0
- Invalid restaurant IDs: 0
- Invalid driver IDs: 0

### Temporal Validation

Hourly order counts confirmed the intended demand patterns.

Lunch demand increased between approximately 11:00 and 14:00, while the
strongest demand occurred during the dinner period from approximately
18:00 to 22:00.

Weekend timestamps were assigned a higher sampling weight through a 1.5× weekend demand multiplier.

The dataset was saved to:

`data/raw/orders.csv`



## Deliveries Dataset

The Deliveries dataset represents the operational outcome of each order.
It connects orders with delivery-related variables such as distance,
preparation time, delivery time and tip amount.

The dataset is designed primarily to support the Day 4 supervised learning
task of predicting delivery time.

### Dataset Size

- Records: 50,000
- Features: 7
- One delivery record is generated for each order in the current synthetic dataset.

### Features

| Feature | Description |
|---|---|
| delivery_id | Unique delivery identifier |
| order_id | Order associated with the delivery |
| delivery_distance_km | Distance between restaurant and customer |
| preparation_time_min | Time required by the restaurant to prepare the order |
| delivery_time_min | Total time taken for the delivery |
| tip_amount | Tip given by the customer |
| delivery_status | Delivery outcome: Delivered or Failed |

### Relationship with Orders

Each delivery is linked to an order through `order_id`.

The synthetic data maintains the following relationship:

- Completed order → Delivered
- Cancelled order → Failed

This maintains consistency between the Orders and Deliveries datasets.

### Delivery Distance

Delivery distance was generated between 0.5 km and 20 km.

Most deliveries are expected to be relatively short, while a smaller
number of deliveries represent longer-distance orders.

Delivery distance is an important predictor for delivery-time prediction
because longer distances generally require more travel time.

### Preparation Time

Preparation time is derived from the restaurant's `avg_prep_time`
and includes small operational variation.

This preserves a relationship between restaurant characteristics and
delivery operations.

The generated preparation time ranged from approximately 5 to 54 minutes.

### Delivery Time

`delivery_time_min` is the primary target variable for the Day 4 regression
task.

Delivery time was generated using multiple operational factors:

- Delivery distance
- Restaurant preparation time
- Traffic condition
- Weather
- Peak ordering hours
- Driver vehicle type
- Random operational variation

The conceptual relationship is:

    Delivery Time
    = Distance Effect
    + Preparation Time
    + Traffic Effect
    + Weather Effect
    + Peak-Hour Effect
    + Vehicle Effect
    + Random Noise

Random variation was intentionally included so that the regression models
would need to learn the underlying relationships rather than reproduce a
perfect deterministic formula.

### Traffic Effect

Traffic conditions from the Orders dataset influence delivery time.

Expected relationship:

    Low Traffic < Medium Traffic < High Traffic

Validation confirmed this relationship:

| Traffic | Average Delivery Time |
|---|---:|
| Low | 44.70 min |
| Medium | 55.37 min |
| High | 66.38 min |

This creates a meaningful operational relationship for the regression model.

### Weather Effect

Weather conditions also influence delivery time.

Expected relationship:

    Clear < Cloudy < Rain < Storm

Validation confirmed this relationship:

| Weather | Average Delivery Time |
|---|---:|
| Clear | 53.37 min |
| Cloudy | 55.17 min |
| Rain | 61.41 min |
| Storm | 68.35 min |

This allows the regression models to learn the effect of adverse weather
on delivery performance.

### Tip Amount

Tip amount was generated primarily as a percentage of the order amount,
with random variation.

This creates a positive relationship between order value and potential
tip amount.

The generated tip amount for delivered orders ranged from approximately
₹2.52 to ₹308.54.

Tip prediction remains an optional secondary modeling problem, while
delivery-time prediction is the primary regression objective.

### Delivery Status

The dataset contains two delivery statuses:

- Delivered
- Failed

The distribution was:

| Status | Records |
|---|---:|
| Delivered | 46,046 |
| Failed | 3,954 |

This exactly matches the Completed/Cancelled distribution in the Orders
dataset.

### Validation

The dataset was validated for:

- Dataset shape
- Duplicate delivery IDs
- Duplicate order IDs
- Missing values
- Delivery distance range
- Preparation time range
- Delivery time range
- Tip amount range
- Delivery status distribution
- Referential integrity with Orders

### Validation Results

- Shape: 50,000 × 7
- Duplicate delivery IDs: 0
- Duplicate order IDs: 0
- Missing values: 0
- Distance range: 0.5–20.0 km
- Preparation time range: 5–54 minutes
- Delivered records: 46,046
- Failed records: 3,954
- Delivery time range for delivered orders: 10.0–143.3 minutes
- Tip amount range for delivered orders: ₹2.52–₹308.54
- Invalid order IDs: 0

### Modeling Consideration

Cancelled/failed deliveries do not represent actual completed delivery
times. Therefore, these records will be excluded from the delivery-time
regression training dataset.

The final regression dataset will primarily use successfully delivered
orders.

### Output

The dataset was saved to:

`data/raw/deliveries.csv`



## Reviews Dataset

The Reviews dataset represents customer feedback associated with completed
orders. It is primarily designed to support the Day 3 NLP and sentiment
analysis module.

The dataset contains both structured information such as ratings and
unstructured natural-language customer feedback.

### Dataset Size

- Records: 30,000
- Features: 8
- Reviews are generated only for completed orders.
- 30,000 completed orders were sampled to receive reviews, representing 60% of all orders and approximately 65% of completed orders.

Approximately 60% of the 50,000 orders were selected to have customer
reviews.

### Features

| Feature | Description |
|---|---|
| review_id | Unique review identifier |
| order_id | Order associated with the review |
| customer_id | Customer who submitted the review |
| restaurant_id | Restaurant associated with the order |
| rating | Customer rating from 1 to 5 |
| review_text | Written customer feedback |
| sentiment | Positive, Negative, or Neutral |
| review_timestamp | Timestamp when the review was submitted |

### Review Eligibility

Reviews were generated only from completed orders.

The synthetic relationship is:

    Completed Order → Possible Customer Review

Cancelled orders were excluded because they do not represent a completed
customer delivery experience.

### Rating Distribution

| Rating | Reviews |
|---:|---:|
| 1 | 1,777 |
| 2 | 2,696 |
| 3 | 4,485 |
| 4 | 10,588 |
| 5 | 10,454 |

The distribution intentionally contains more 4- and 5-star reviews while
still retaining lower ratings for negative customer experiences.

### Sentiment Distribution

| Sentiment | Reviews |
|---|---:|
| Positive | 20,858 |
| Negative | 5,700 |
| Neutral | 3,442 |

The dataset contains all three sentiment classes required for supervised
classification.

### Relationship Between Rating and Sentiment

Ratings and sentiment were generated with a logical relationship while
allowing a small amount of inconsistency to make the synthetic data more
realistic.

General relationship:

    4–5 stars → Mostly Positive
    3 stars   → Mostly Neutral
    1–2 stars → Mostly Negative

Validation produced the following average ratings:

| Sentiment | Average Rating |
|---|---:|
| Negative | 2.23 |
| Neutral | 3.00 |
| Positive | 4.42 |

This confirms that the sentiment labels are directionally consistent with
customer ratings.

### Review Themes

Review text was designed around common food-delivery experiences.

#### Positive Themes

- Delicious food
- Fresh food
- Fast delivery
- Hot food
- Good packaging
- Good portions
- Excellent service
- Good quality

#### Negative Themes

- Late delivery
- Cold food
- Long delivery time
- Bland food
- Missing items
- Wrong items
- Damaged packaging
- Poor delivery experience

#### Neutral Themes

- Average experience
- Acceptable food
- Nothing special
- Decent food
- Satisfactory experience

These themes are intentionally included so that traditional NLP techniques
such as TF-IDF can identify meaningful vocabulary associated with each
sentiment class.

### NLP Objective

The Reviews dataset will be used to build a sentiment classification
pipeline:

    Raw Review Text
          ↓
    Text Cleaning
          ↓
    Tokenization
          ↓
    Stop-word Removal
          ↓
    Lemmatization
          ↓
    TF-IDF
          ↓
    Sentiment Classification
          ↓
    Model Evaluation

The project will compare traditional machine-learning approaches such as
Naive Bayes and SVM.

### Business Objective

The NLP analysis will be used to answer questions such as:

- What percentage of customer feedback is negative?
- What words and phrases are associated with negative experiences?
- What are the major customer complaint themes?
- Are complaints primarily related to food, delivery, packaging, or order
  accuracy?

The objective is therefore not only sentiment classification but also
identifying operational areas that require improvement.

### Review Timestamp

Review timestamps were generated from the corresponding order timestamp
with a delay of approximately 1–48 hours.

Therefore, reviews may extend beyond the order dataset's June 30 end date.

Validation showed the review period as:

    2026-01-01 to 2026-07-02

### Validation

The dataset was validated for:

- Dataset shape
- Duplicate review IDs
- Duplicate order IDs
- Missing values
- Rating range
- Rating distribution
- Sentiment distribution
- Rating-sentiment relationship
- Review date range
- Referential integrity

### Validation Results

- Shape: 30,000 × 8
- Duplicate review IDs: 0
- Duplicate order IDs: 0
- Missing values: 0
- Rating range: 1–5
- Invalid order IDs: 0
- Invalid customer IDs: 0
- Invalid restaurant IDs: 0
- Negative reviews: 5,700
- Neutral reviews: 3,442
- Positive reviews: 20,858

### Output

The dataset was saved to:

`data/raw/reviews.csv`



## Payments Dataset

The Payments dataset represents payment transactions associated with
customer orders.

It provides financial and transaction-level information that can be used
for revenue analysis, payment-method analysis, refund analysis and
business intelligence.

Although Payments is not the primary input to one of the mandatory ML
models, it provides additional business context for the end-to-end Uber
Eats intelligence system.

### Dataset Size

- Records: 50,000
- Features: 7
- One payment record is associated with each order.

### Features

| Feature | Description |
|---|---|
| payment_id | Unique payment transaction identifier |
| order_id | Order associated with the payment |
| customer_id | Customer making the payment |
| payment_method | Method used for payment |
| payment_status | Successful, Failed, or Refunded |
| amount_paid | Amount processed for the payment transaction; ₹0 for failed payments |
| payment_timestamp | Timestamp of the payment transaction |

### Payment Methods

The dataset contains five payment methods:

- UPI
- Credit Card
- Debit Card
- Wallet
- Cash

The distribution was generated using different probabilities to represent
different levels of adoption across payment methods.

### Payment Status

The dataset contains three payment outcomes:

- Successful
- Failed
- Refunded

The payment status is logically related to the corresponding order status.

Completed orders can have successful or refunded payments, while cancelled orders can have failed or refunded payments.

For completed orders, 97% of payments are generated as Successful and 3% as Refunded. For cancelled orders, 70% are generated as Failed and 30% as Refunded.

### Amount Logic

For successful and refunded transactions:

    amount_paid = order_amount

For failed transactions:

    amount_paid = 0

This allows payment failures to be distinguished from transactions where
money was successfully processed.

### Payment Distribution

| Payment Method | Records |
|---|---:|
| UPI | 19,794 |
| Credit Card | 12,530 |
| Debit Card | 7,511 |
| Wallet | 6,080 |
| Cash | 4,085 |

### Payment Status Distribution

| Status | Records |
|---|---:|
| Successful | 44,610 |
| Failed | 2,755 |
| Refunded | 2,635 |

### Average Amount by Payment Status

| Status | Average Amount |
|---|---:|
| Failed | ₹0.00 |
| Refunded | ₹575.92 |
| Successful | ₹567.01 |

### Business Use Cases

The Payments dataset can support business questions such as:

- Which payment methods are most frequently used?
- What is the payment failure rate?
- What percentage of transactions are refunded?
- How much transaction value is associated with successful payments?
- Which customer segments generate the highest transaction value?
- Are particular payment methods associated with higher failure rates?

These analyses can be incorporated into the business-insights stage of
the project.

### Validation

The dataset was validated for:

- Dataset shape
- Duplicate payment IDs
- Duplicate order IDs
- Missing values
- Payment amount range
- Payment-method distribution
- Payment-status distribution
- Average amount by payment status
- Failed payment amount consistency
- Referential integrity

### Validation Results

- Shape: 50,000 × 7
- Duplicate payment IDs: 0
- Duplicate order IDs: 0
- Missing values: 0
- Amount range: ₹0.00–₹2,099.80
- Failed payments with amount > ₹0: 0
- Invalid order IDs: 0
- Invalid customer IDs: 0

### Output

The dataset was saved to:

`data/raw/payments.csv`



## Cross-Dataset Validation

After generating all seven datasets, a final cross-dataset validation was
performed to verify that the synthetic data is internally consistent.

### Dataset Sizes

| Dataset | Records | Features |
|---|---:|---:|
| Customers | 5,000 | 5 |
| Restaurants | 300 | 6 |
| Drivers | 500 | 5 |
| Orders | 50,000 | 11 |
| Deliveries | 50,000 | 7 |
| Reviews | 30,000 | 8 |
| Payments | 50,000 | 7 |

### Referential Integrity

The following relationships were validated:

- Every customer referenced by an order exists in the Customers dataset.
- Every restaurant referenced by an order exists in the Restaurants dataset.
- Every driver referenced by an order exists in the Drivers dataset.
- Every delivery references a valid order.
- Every review references a valid order.
- Every payment references a valid order.

All referential-integrity checks returned zero invalid records.

### Business Logic Validation

The following business rules were validated:

- Every completed order has a Delivered delivery record.
- Every cancelled order has a Failed delivery record.
- Reviews are associated only with completed orders.

| Business Rule                             | Violations |
| ----------------------------------------- | ---------: |
| Completed orders without Delivered record |          0 |
| Cancelled orders without Failed delivery  |          0 |
| Reviews belonging to non-completed orders |          0 |

All business-logic checks returned zero violations.

### Duplicate Validation

All primary identifiers were checked for duplicates:

- Customer IDs
- Restaurant IDs
- Driver IDs
- Order IDs
- Delivery IDs
- Review IDs
- Payment IDs

All duplicate checks returned zero duplicates.

### Final Validation Result

The complete synthetic data system passed all cross-dataset validation
checks.

This confirms that the datasets can be used together for downstream
feature engineering, machine learning, forecasting and business analysis.