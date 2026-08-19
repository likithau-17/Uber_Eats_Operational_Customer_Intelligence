# Customer Feature Engineering

## 1. Objective

The objective of this module is to transform raw Uber Eats customer, order, and review data into a **customer-level behavioral feature dataset**.

The resulting dataset is used as the input for **customer segmentation**.

The main goal is to represent each customer using measurable behavioral characteristics such as:

* Order volume
* Spending
* Average order value
* Ordering frequency
* Weekend ordering behavior
* Late-night ordering behavior
* Rating behavior

---

## 2. Input Datasets

The feature engineering process uses three raw datasets.

### Customers

Used to obtain the complete customer population and `customer_id`.

### Orders

Used to calculate customer purchase and ordering behavior.

Important columns include:

* `order_id`
* `customer_id`
* `order_timestamp`
* `order_amount`
* `order_status`

### Reviews

Used to calculate the average rating given by each customer.

Important columns include:

* `customer_id`
* `rating`
* `review_timestamp`

---

## 3. Completed Order Filtering

Only completed orders are used for purchase-behavior features.

```python
completed_orders = orders[
    orders["order_status"] == "Completed"
].copy()
```

This ensures that cancelled or otherwise incomplete orders do not contribute to customer purchase behavior.

---

## 4. Customer Behavioral Features

The following seven features were created to represent different dimensions of customer behavior.

### 4.1 `total_orders`

**Definition:** Total number of completed orders placed by a customer.

**Calculation:**

```text
total_orders = count of completed order_id
```

This represents the customer's overall ordering activity.

---

### 4.2 `avg_order_value`

**Definition:** Average amount spent per completed order.

**Calculation:**

```text
avg_order_value = mean(order_amount)
```

This represents the customer's typical spending per order.

---

### 4.3 `total_spending`

**Definition:** Total amount spent by a customer across all completed orders.

**Calculation:**

```text
total_spending = sum(order_amount)
```

This represents the customer's overall monetary value.

---

### 4.4 `ordering_frequency`

**Definition:** Average number of completed orders placed per active month.

**Calculation:**

```text
ordering_frequency = total_orders / active_months
```

`active_months` represents the number of distinct months in which the customer placed a completed order.

`active_months` is used to calculate ordering frequency but is **not retained as a clustering feature**.

---

### 4.5 `avg_rating_given`

**Definition:** Average rating provided by the customer across their reviews.

**Calculation:**

```text
avg_rating_given = mean(rating)
```

This represents the customer's rating behavior.

Customers without review activity are retained in the dataset. Their missing rating value is represented as `0` and documented as **no review activity**.

---

### 4.6 `weekend_orders`

**Definition:** Number of completed orders placed on Saturday or Sunday.

The order timestamp is used to identify the day of the week.

* Saturday = `5`
* Sunday = `6`

This captures the customer's weekend ordering behavior.

---

### 4.7 `late_night_orders`

**Definition:** Number of completed orders placed during the defined late-night period.

The project defines late-night orders as orders placed:

* At or after `22:00`
* Before `04:00`

This captures the customer's late-night ordering pattern.

---

## 5. Feature Summary

| Feature              | Represents                         |
| -------------------- | ---------------------------------- |
| `total_orders`       | Overall ordering activity          |
| `avg_order_value`    | Typical spending per order         |
| `total_spending`     | Overall customer monetary value    |
| `ordering_frequency` | Ordering activity per active month |
| `avg_rating_given`   | Customer rating behavior           |
| `weekend_orders`     | Weekend ordering behavior          |
| `late_night_orders`  | Late-night ordering behavior       |

These features collectively capture **customer engagement, spending behavior, ordering frequency, and ordering patterns**.

They are used as inputs for the **customer segmentation model**.

---

## 6. Feature Engineering Output

The feature engineering process produces a **customer-level dataset**, where each row represents one unique customer.

### Output File

```text
data/processed/customer_features.csv
```

### Output Columns

The resulting dataset contains:

```text
customer_id
total_orders
avg_order_value
total_spending
weekend_orders
late_night_orders
ordering_frequency
avg_rating_given
```

The `customer_id` column is retained for customer identification and for linking customers to their segmentation results.

However, `customer_id` is **not used as a machine learning feature**.

---

## 7. Handling Customers Without Completed Orders

The complete customer population is retained during feature engineering.

A `left join` is used when combining customer information with order-based features:

```python
customer_features = customer_features.merge(
    order_features,
    on="customer_id",
    how="left"
)
```

This ensures that customers without completed orders are not removed from the dataset.

For customers without completed orders, the order-based behavioral features are filled with `0`.

This allows the final dataset to represent the **complete customer population**.

---

## 8. Handling Customers Without Reviews

Customers without review activity are also retained.

Their `avg_rating_given` value is filled with `0`:

```python
customer_features["avg_rating_given"] = (
    customer_features["avg_rating_given"].fillna(0)
)
```

In this dataset, `0` represents **no review activity**, rather than an actual rating of zero.

This distinction should be considered when interpreting the feature during customer profiling.

---

## 9. Data Type Handling

Count-based features are converted to integer data types:

* `total_orders`
* `weekend_orders`
* `late_night_orders`

Continuous behavioral features remain numeric:

* `avg_order_value`
* `total_spending`
* `ordering_frequency`
* `avg_rating_given`

This ensures that the resulting feature matrix is suitable for statistical analysis and machine learning.

---

## 10. Feature Validation

The generated feature dataset was validated before being passed to the segmentation module.

The following checks were performed:

* Customer count
* Duplicate customer IDs
* Missing values
* Expected feature columns
* Numeric feature values

The final feature dataset contains:

```text
5,000 customers
8 columns
```

The dataset contains **no missing values** after the defined missing-value treatment.

Duplicate customer IDs were also checked to ensure that each customer appears only once.

---

## 11. Feature Design

The selected features represent several important dimensions of customer behavior.

### Customer Engagement

```text
total_orders
ordering_frequency
```

These features measure how frequently customers use the platform.

### Customer Monetary Value

```text
avg_order_value
total_spending
```

These features represent customer spending behavior and overall monetary contribution.

### Ordering Patterns

```text
weekend_orders
late_night_orders
```

These features capture when customers tend to place orders.

### Customer Rating Behavior

```text
avg_rating_given
```

This captures the customer's tendency to provide ratings and their average rating behavior.

---

## 12. Excluding Identifiers from Clustering

The `customer_id` column is removed before machine learning:

```python
feature_matrix = df.drop(
    columns=["customer_id"]
)
```

Identifiers should not be used as clustering features because they do not represent meaningful customer behavior.

Using identifiers could introduce arbitrary numerical relationships into distance-based algorithms such as:

* K-Means
* DBSCAN

Therefore:

```text
customer_id → Identification only
behavioral features → Machine learning inputs
```

---

## 13. Why Feature Engineering Is Required

The raw orders dataset contains **one row per order**.

A single customer can therefore appear across many rows:

```text
Customer A → Order 1
Customer A → Order 2
Customer A → Order 3
Customer A → Order 4
```

Clustering requires **one feature vector per customer**.

Feature engineering transforms transactional data into a customer-level representation:

```text
Customer A → [orders, spending, frequency, weekend behavior, ...]
Customer B → [orders, spending, frequency, weekend behavior, ...]
Customer C → [orders, spending, frequency, weekend behavior, ...]
```

This customer-level behavioral matrix becomes the input to the segmentation process.

---

## 14. Feature Engineering Workflow

The complete feature engineering process follows this sequence:

```text
Raw Customer Data
        │
        ├──────────────────┐
        │                  │
        ▼                  ▼
   Order Data          Review Data
        │                  │
        ▼                  ▼
Filter Completed      Calculate Average
Orders                Customer Rating
        │                  │
        ▼                  │
Create Behavioral        │
Features                 │
        │                  │
        └─────────┬────────┘
                  ▼
          Merge by customer_id
                  │
                  ▼
          Handle Missing Values
                  │
                  ▼
           Validate Features
                  │
                  ▼
       customer_features.csv
                  │
                  ▼
        Customer Segmentation
```

---

## 15. Connection to Segmentation

The generated feature dataset is passed to the customer segmentation module:

```text
data/processed/customer_features.csv
                  │
                  ▼
        src/segmentation.py
                  │
                  ├── Customer EDA
                  │
                  ├── StandardScaler
                  │
                  ├── K-Means
                  │
                  ├── DBSCAN
                  │
                  └── PCA
```

The feature engineering module therefore acts as the **bridge between the raw transactional datasets and the machine learning segmentation workflow**.

---

## 16. Limitations

The current feature set is intentionally focused on core customer purchasing behavior.

Some potentially useful behavioral dimensions are not included in the first version.

For example:

* Recency of last order
* Customer lifetime value
* Number of unique restaurants visited
* Cuisine/category diversity
* Average delivery distance
* Cancellation behavior
* Discount or promotion usage
* Payment method preferences
* Customer tenure
* Review sentiment
* Repeat ordering from the same restaurant

These features could be incorporated in future versions to create a more comprehensive customer representation.

---

## 17. Future Improvements

Future versions of the feature engineering pipeline could include additional behavioral metrics such as:

### Recency

Number of days since the customer's most recent order.

### Customer Lifetime Value

Estimated monetary value generated by the customer over their relationship with the platform.

### Restaurant Diversity

Number of unique restaurants ordered from.

### Cuisine Diversity

Number of different restaurant categories or cuisines ordered from.

### Cancellation Behavior

Customer-level cancellation frequency and cancellation rate.

### Promotion Behavior

Frequency of using discounts, coupons, or promotional offers.

These additional features could improve the ability of clustering algorithms to distinguish between different customer behavior patterns.

---

## 18. Final Output

The final feature-engineered dataset is:

```text
data/processed/customer_features.csv
```

It contains **5,000 unique customers** represented by seven behavioral features plus the customer identifier.

The dataset is now ready for the next stage:

**Customer Segmentation using K-Means, DBSCAN, and PCA.**

---

## 19. Summary

The feature engineering module transforms raw Uber Eats-style transactional and review data into a structured **customer-level behavioral dataset**.

The resulting features capture:

* Ordering activity
* Spending behavior
* Ordering frequency
* Weekend ordering behavior
* Late-night ordering behavior
* Rating behavior

The final dataset provides the **behavioral foundation required for customer segmentation and downstream business analysis**.
