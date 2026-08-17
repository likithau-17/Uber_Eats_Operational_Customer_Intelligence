import pandas as pd


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

customers = pd.read_csv(
    "data/raw/customers.csv"
)

restaurants = pd.read_csv(
    "data/raw/restaurants.csv"
)

drivers = pd.read_csv(
    "data/raw/drivers.csv"
)

orders = pd.read_csv(
    "data/raw/orders.csv"
)

deliveries = pd.read_csv(
    "data/raw/deliveries.csv"
)

reviews = pd.read_csv(
    "data/raw/reviews.csv"
)

payments = pd.read_csv(
    "data/raw/payments.csv"
)


# --------------------------------------------------
# Dataset sizes
# --------------------------------------------------

print("\n========== DATASET SIZES ==========")

datasets = {
    "Customers": customers,
    "Restaurants": restaurants,
    "Drivers": drivers,
    "Orders": orders,
    "Deliveries": deliveries,
    "Reviews": reviews,
    "Payments": payments
}

for name, df in datasets.items():

    print(
        f"{name}: {df.shape}"
    )


# --------------------------------------------------
# Referential Integrity
# --------------------------------------------------

print("\n========== REFERENTIAL INTEGRITY ==========")


invalid_customer_orders = (
    ~orders["customer_id"].isin(
        customers["customer_id"]
    )
).sum()

print(
    "Invalid customer IDs in Orders:",
    invalid_customer_orders
)


invalid_restaurant_orders = (
    ~orders["restaurant_id"].isin(
        restaurants["restaurant_id"]
    )
).sum()

print(
    "Invalid restaurant IDs in Orders:",
    invalid_restaurant_orders
)


invalid_driver_orders = (
    ~orders["driver_id"].isin(
        drivers["driver_id"]
    )
).sum()

print(
    "Invalid driver IDs in Orders:",
    invalid_driver_orders
)


invalid_delivery_orders = (
    ~deliveries["order_id"].isin(
        orders["order_id"]
    )
).sum()

print(
    "Invalid order IDs in Deliveries:",
    invalid_delivery_orders
)


invalid_review_orders = (
    ~reviews["order_id"].isin(
        orders["order_id"]
    )
).sum()

print(
    "Invalid order IDs in Reviews:",
    invalid_review_orders
)


invalid_payment_orders = (
    ~payments["order_id"].isin(
        orders["order_id"]
    )
).sum()

print(
    "Invalid order IDs in Payments:",
    invalid_payment_orders
)


# --------------------------------------------------
# Business Logic Checks
# --------------------------------------------------

print("\n========== BUSINESS LOGIC ==========")


# Completed orders should be Delivered
completed_orders = orders[
    orders["order_status"] == "Completed"
]

delivered_orders = deliveries[
    deliveries["delivery_status"] == "Delivered"
]

completed_not_delivered = (
    ~completed_orders["order_id"].isin(
        delivered_orders["order_id"]
    )
).sum()

print(
    "Completed orders without Delivered record:",
    completed_not_delivered
)


# Cancelled orders should be Failed
cancelled_orders = orders[
    orders["order_status"] == "Cancelled"
]

failed_deliveries = deliveries[
    deliveries["delivery_status"] == "Failed"
]

cancelled_not_failed = (
    ~cancelled_orders["order_id"].isin(
        failed_deliveries["order_id"]
    )
).sum()

print(
    "Cancelled orders without Failed delivery:",
    cancelled_not_failed
)


# Reviews should belong to completed orders
review_order_status = reviews.merge(
    orders[
        [
            "order_id",
            "order_status"
        ]
    ],
    on="order_id",
    how="left"
)

reviews_for_cancelled_orders = (
    review_order_status["order_status"]
    != "Completed"
).sum()

print(
    "Reviews belonging to non-completed orders:",
    reviews_for_cancelled_orders
)


# --------------------------------------------------
# Duplicate Checks
# --------------------------------------------------

print("\n========== DUPLICATE CHECKS ==========")

print(
    "Duplicate Customer IDs:",
    customers["customer_id"].duplicated().sum()
)

print(
    "Duplicate Restaurant IDs:",
    restaurants["restaurant_id"].duplicated().sum()
)

print(
    "Duplicate Driver IDs:",
    drivers["driver_id"].duplicated().sum()
)

print(
    "Duplicate Order IDs:",
    orders["order_id"].duplicated().sum()
)

print(
    "Duplicate Delivery IDs:",
    deliveries["delivery_id"].duplicated().sum()
)

print(
    "Duplicate Review IDs:",
    reviews["review_id"].duplicated().sum()
)

print(
    "Duplicate Payment IDs:",
    payments["payment_id"].duplicated().sum()
)


# --------------------------------------------------
# Final Summary
# --------------------------------------------------

print("\n========== VALIDATION COMPLETE ==========")

print(
    "All datasets loaded successfully."
)