from pathlib import Path

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def create_customer_features():
    """
    Create a customer-level behavioral feature table for segmentation.

    Features created:
        - total_orders
        - avg_order_value
        - total_spending
        - ordering_frequency
        - avg_rating_given
        - weekend_orders
        - late_night_orders

    Business rule:
        Cancelled orders are excluded from purchase-behavior metrics
        because they do not represent completed purchases.
    """

    customers = pd.read_csv(
        RAW_DATA_DIR / "customers.csv",
        parse_dates=["signup_date"]
    )

    orders = pd.read_csv(
        RAW_DATA_DIR / "orders.csv",
        parse_dates=["order_timestamp"]
    )

    reviews = pd.read_csv(
        RAW_DATA_DIR / "reviews.csv",
        parse_dates=["review_timestamp"]
    )

    completed_orders = orders[
        orders["order_status"] == "Completed"
    ].copy()

    completed_orders["order_month"] = (
        completed_orders["order_timestamp"].dt.to_period("M")
    )

    completed_orders["order_day_of_week"] = (
        completed_orders["order_timestamp"].dt.dayofweek
    )

    completed_orders["order_hour"] = (
        completed_orders["order_timestamp"].dt.hour
    )

    completed_orders["is_weekend"] = (
        completed_orders["order_day_of_week"].isin([5, 6])
    )

    completed_orders["is_late_night"] = (
        (completed_orders["order_hour"] >= 22)
        | (completed_orders["order_hour"] < 4)
    )

    order_features = (
        completed_orders
        .groupby("customer_id")
        .agg(
            total_orders=("order_id", "count"),
            avg_order_value=("order_amount", "mean"),
            total_spending=("order_amount", "sum"),
            active_months=("order_month", "nunique"),
            weekend_orders=("is_weekend", "sum"),
            late_night_orders=("is_late_night", "sum"),
        )
        .reset_index()
    )

    order_features["ordering_frequency"] = (
        order_features["total_orders"]
        / order_features["active_months"]
    )

    # active_months is no longer needed as a clustering feature.
    order_features = order_features.drop(columns=["active_months"])

    rating_features = (
        reviews
        .groupby("customer_id")
        .agg(
            avg_rating_given=("rating", "mean")
        )
        .reset_index()
    )

    customer_features = customers[["customer_id"]].copy()

    customer_features = customer_features.merge(
        order_features,
        on="customer_id",
        how="left"
    )

    customer_features = customer_features.merge(
        rating_features,
        on="customer_id",
        how="left"
    )
    order_columns = [
        "total_orders",
        "avg_order_value",
        "total_spending",
        "ordering_frequency",
        "weekend_orders",
        "late_night_orders",
    ]

    customer_features[order_columns] = (
        customer_features[order_columns].fillna(0)
    )

    # Customers with no reviews are retained.
    # For the first clustering version, missing rating behavior is
    # represented as 0 and documented as "no review activity".
    customer_features["avg_rating_given"] = (
        customer_features["avg_rating_given"].fillna(0)
    )

    customer_features["total_orders"] = (
        customer_features["total_orders"].astype(int)
    )

    customer_features["weekend_orders"] = (
        customer_features["weekend_orders"].astype(int)
    )

    customer_features["late_night_orders"] = (
        customer_features["late_night_orders"].astype(int)
    )

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        PROCESSED_DATA_DIR / "customer_features.csv"
    )

    customer_features.to_csv(
        output_path,
        index=False
    )

    print("\n========== CUSTOMER FEATURE ENGINEERING ==========")
    print(f"Customers: {len(customer_features):,}")
    print(f"Completed orders used: {len(completed_orders):,}")
    print(f"Output shape: {customer_features.shape}")
    print(f"Output saved to: {output_path}")

    print("\nFeature columns:")
    print(customer_features.columns.tolist())

    print("\nFeature preview:")
    print(customer_features.head())

    return customer_features


if __name__ == "__main__":
    create_customer_features()