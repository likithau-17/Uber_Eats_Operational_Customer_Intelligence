# Uber Eats Operational & Customer Intelligence System
## Data Dictionary

This document describes the structure, meaning and role of each dataset
used in the Uber Eats Operational & Customer Intelligence System.

The project uses synthetic data generated for machine-learning and
business-analytics purposes.

---

# 1. Customers

**File:** `data/raw/customers.csv`

| Column | Data Type | Description | Role |
|---|---|---|---|
| customer_id | String | Unique customer identifier | Primary Key |
| age | Integer | Customer age | Feature |
| gender | String | Customer gender | Feature |
| city | String | Customer's city | Feature |
| signup_date | Datetime | Date and time when customer registered | Feature |

**Records:** 5,000

---

# 2. Restaurants

**File:** `data/raw/restaurants.csv`

| Column | Data Type | Description | Role |
|---|---|---|---|
| restaurant_id | String | Unique restaurant identifier | Primary Key |
| restaurant_name | String | Restaurant name | Descriptive |
| restaurant_category | String | Cuisine/category of restaurant | Feature |
| rating | Float | Restaurant rating | Feature |
| avg_prep_time | Integer | Average food preparation time in minutes | Feature |
| city | String | Restaurant location | Feature |

**Records:** 300

---

# 3. Drivers

**File:** `data/raw/drivers.csv`

| Column | Data Type | Description | Role |
|---|---|---|---|
| driver_id | String | Unique driver identifier | Primary Key |
| vehicle_type | String | Driver's delivery vehicle | Feature |
| driver_rating | Float | Driver rating | Feature |
| experience_years | Integer | Driver's delivery experience | Feature |
| city | String | Driver's operating city | Feature |

**Records:** 500

---

# 4. Orders

**File:** `data/raw/orders.csv`

| Column | Data Type | Description | Role |
|---|---|---|---|
| order_id | String | Unique order identifier | Primary Key |
| customer_id | String | Customer placing the order | Foreign Key |
| restaurant_id | String | Restaurant receiving the order | Foreign Key |
| driver_id | String | Driver assigned to the order | Foreign Key |
| order_timestamp | Datetime | Time when the order was placed | Feature |
| order_amount | Float | Total order value | Feature |
| num_items | Integer | Number of items in the order | Feature |
| order_status | String | Completed or Cancelled | Target/Feature |
| city | String | Order location | Feature |
| weather | String | Weather condition during the order | Feature |
| traffic_condition | String | Traffic condition during the order | Feature |

**Records:** 50,000

---

# 5. Deliveries

**File:** `data/raw/deliveries.csv`

| Column | Data Type | Description | Role |
|---|---|---|---|
| delivery_id | String | Unique delivery identifier | Primary Key |
| order_id | String | Associated order | Foreign Key |
| delivery_distance_km | Float | Delivery distance in kilometers | Feature |
| preparation_time_min | Float | Restaurant preparation time | Feature |
| delivery_time_min | Float | Total delivery time in minutes | Regression Target |
| tip_amount | Float | Customer tip amount | Regression Target/Feature |
| delivery_status | String | Delivered or Failed | Feature |

**Records:** 50,000

**Primary ML target:**

`delivery_time_min`

---

# 6. Reviews

**File:** `data/raw/reviews.csv`

| Column | Data Type | Description | Role |
|---|---|---|---|
| review_id | String | Unique review identifier | Primary Key |
| order_id | String | Associated order | Foreign Key |
| customer_id | String | Customer submitting the review | Foreign Key |
| restaurant_id | String | Restaurant associated with the review | Foreign Key |
| rating | Integer | Customer rating from 1 to 5 | Feature |
| review_text | String | Written customer feedback | NLP Input |
| sentiment | String | Positive, Neutral or Negative | NLP Target |
| review_timestamp | Datetime | Time when review was submitted | Feature |

**Records:** 30,000

**Primary NLP input:**

`review_text`

**Primary NLP target:**

`sentiment`

---

# 7. Payments

**File:** `data/raw/payments.csv`

| Column | Data Type | Description | Role |
|---|---|---|---|
| payment_id | String | Unique payment identifier | Primary Key |
| order_id | String | Associated order | Foreign Key |
| customer_id | String | Customer making payment | Foreign Key |
| payment_method | String | Payment method used | Feature |
| payment_status | String | Successful, Failed or Refunded | Feature |
| amount_paid | Float | Payment transaction amount | Feature |
| payment_timestamp | Datetime | Time of payment | Feature |

**Records:** 50,000

---

# Dataset Relationships

The datasets are connected through shared identifiers.

```text
Customers
    │
    │ customer_id
    ▼
Orders ◄──────── Restaurants
    │                 │
    │                 │ restaurant_id
    │
    ├──────────────► Deliveries
    │
    ├──────────────► Reviews
    │
    └──────────────► Payments

Drivers
    │
    │ driver_id
    ▼
Orders