import numpy as np
import pandas as pd
from pathlib import Path


RANDOM_SEED = 42
NUM_CUSTOMERS = 5_000
NUM_RESTAURANTS = 300
NUM_DRIVERS = 500
NUM_ORDERS = 50_000
NUM_DELIVERIES = 50_000
NUM_REVIEWS = 30_000
NUM_PAYMENTS = 50_000

np.random.seed(RANDOM_SEED)


def generate_customers(num_customers=NUM_CUSTOMERS):
    """Generate synthetic customer data."""

    customer_ids = [
        f"C{str(i).zfill(5)}"
        for i in range(1, num_customers + 1)
    ]

    ages = np.random.randint(18, 66, size=num_customers)

    genders = np.random.choice(
        ["Male", "Female"],
        size=num_customers,
        p=[0.5, 0.5]
    )

    cities = np.random.choice(
        [
            "Bangalore",
            "Hyderabad",
            "Chennai",
            "Mumbai",
            "Delhi",
            "Pune",
            "Kolkata"
        ],
        size=num_customers
    )

    signup_dates = pd.to_datetime(
        np.random.randint(
            pd.Timestamp("2025-01-01").value // 10**9,
            pd.Timestamp("2026-06-30").value // 10**9,
            size=num_customers
        ),
        unit="s"
    )

    customers = pd.DataFrame({
        "customer_id": customer_ids,
        "age": ages,
        "gender": genders,
        "city": cities,
        "signup_date": signup_dates
    })

    return customers


def generate_restaurants(num_restaurants=NUM_RESTAURANTS):
    """Generate synthetic restaurant data."""

    restaurant_ids = [
        f"R{str(i).zfill(4)}"
        for i in range(1, num_restaurants + 1)
    ]

    restaurant_names = [
        "Spice Route",
        "Urban Bites",
        "Curry House",
        "Food Junction",
        "Biryani Bowl",
        "The Tasty Corner",
        "Flavour Hub",
        "Green Plate",
        "Street Kitchen",
        "The Food Studio"
    ]

    categories = [
        "Indian",
        "Chinese",
        "Fast Food",
        "Biryani",
        "Pizza",
        "South Indian",
        "Desserts",
        "Cafe",
        "Bakery",
        "Healthy"
    ]

    cities = [
        "Bangalore",
        "Hyderabad",
        "Chennai",
        "Mumbai",
        "Delhi",
        "Pune",
        "Kolkata"
    ]

    selected_categories = np.random.choice(
        categories,
        size=num_restaurants
    )

    selected_cities = np.random.choice(
        cities,
        size=num_restaurants
    )

    restaurant_name_values = [
        f"{np.random.choice(restaurant_names)} {i}"
        for i in range(1, num_restaurants + 1)
    ]

    restaurant_ratings = np.round(
        np.clip(
            np.random.normal(
                loc=4.1,
                scale=0.4,
                size=num_restaurants
            ),
            3.0,
            5.0
        ),
        1
    )

    prep_time_ranges = {
        "Indian": (20, 40),
        "Chinese": (20, 35),
        "Fast Food": (10, 25),
        "Biryani": (30, 50),
        "Pizza": (20, 35),
        "South Indian": (15, 30),
        "Desserts": (10, 20),
        "Cafe": (10, 25),
        "Bakery": (10, 20),
        "Healthy": (15, 30)
    }

    preparation_times = []

    for category in selected_categories:

        minimum, maximum = prep_time_ranges[category]

        preparation_times.append(
            np.random.randint(
                minimum,
                maximum + 1
            )
        )

    restaurants = pd.DataFrame({
        "restaurant_id": restaurant_ids,
        "restaurant_name": restaurant_name_values,
        "restaurant_category": selected_categories,
        "restaurant_rating": restaurant_ratings,
        "avg_prep_time": preparation_times,
        "city": selected_cities
    })

    return restaurants


def generate_drivers(num_drivers=NUM_DRIVERS):
    """Generate synthetic driver data."""

    driver_ids = [
        f"D{str(i).zfill(4)}"
        for i in range(1, num_drivers + 1)
    ]

    cities = [
        "Bangalore",
        "Hyderabad",
        "Chennai",
        "Mumbai",
        "Delhi",
        "Pune",
        "Kolkata"
    ]

    vehicle_types = np.random.choice(
        ["Bike", "Scooter", "Car"],
        size=num_drivers,
        p=[0.55, 0.30, 0.15]
    )

    driver_ratings = np.round(
        np.clip(
            np.random.normal(
                loc=4.2,
                scale=0.35,
                size=num_drivers
            ),
            3.0,
            5.0
        ),
        1
    )

    experience_years = np.random.choice(
        np.arange(0, 11),
        size=num_drivers,
        p=[
            0.12,
            0.15,
            0.14,
            0.13,
            0.11,
            0.10,
            0.08,
            0.06,
            0.04,
            0.04,
            0.03
        ]
    )

    selected_cities = np.random.choice(
        cities,
        size=num_drivers
    )

    drivers = pd.DataFrame({
        "driver_id": driver_ids,
        "vehicle_type": vehicle_types,
        "driver_rating": driver_ratings,
        "experience_years": experience_years,
        "city": selected_cities
    })

    return drivers


def generate_orders(
    customers,
    restaurants,
    drivers,
    num_orders=NUM_ORDERS
):
    """Generate synthetic order transaction data."""

    cities = customers["city"].unique()

    order_ids = [
        f"O{str(i).zfill(6)}"
        for i in range(1, num_orders + 1)
    ]

    selected_customers = np.random.choice(
        customers["customer_id"],
        size=num_orders
    )

    customer_lookup = customers.set_index("customer_id")

    order_cities = customer_lookup.loc[
        selected_customers,
        "city"
    ].to_numpy()

    restaurant_by_city = {
        city: restaurants.loc[
            restaurants["city"] == city,
            "restaurant_id"
        ].to_numpy()
        for city in cities
    }

    selected_restaurants = [
        np.random.choice(
            restaurant_by_city[city]
        )
        for city in order_cities
    ]

    driver_by_city = {
        city: drivers.loc[
            drivers["city"] == city,
            "driver_id"
        ].to_numpy()
        for city in cities
    }

    selected_drivers = [
        np.random.choice(
            driver_by_city[city]
        )
        for city in order_cities
    ]

    start_date = pd.Timestamp("2026-01-01")
    end_date = pd.Timestamp("2026-06-30 23:59:59")

    timestamps = pd.date_range(
        start=start_date,
        end=end_date,
        freq="h"
    )

    hours = timestamps.hour
    weekdays = timestamps.dayofweek

    # Demand weights by hour
    hour_weights = np.ones(len(timestamps))

    hour_weights[
        (hours >= 11) & (hours <= 14)
    ] = 2.0

    hour_weights[
        (hours >= 18) & (hours <= 22)
    ] = 3.0

    hour_weights[
        (hours >= 23) | (hours <= 1)
    ] = 1.5

    # Weekend multiplier
    weekend_multiplier = np.where(
        weekdays >= 5,
        1.5,
        1.0
    )

    timestamp_weights = (
        hour_weights * weekend_multiplier
    )

    timestamp_weights = (
        timestamp_weights /
        timestamp_weights.sum()
    )

    order_timestamps = np.random.choice(
        timestamps,
        size=num_orders,
        p=timestamp_weights
    )

    order_timestamps = pd.to_datetime(
        order_timestamps
    )

    num_items = np.random.choice(
        [1, 2, 3, 4, 5, 6],
        size=num_orders,
        p=[0.25, 0.30, 0.20, 0.12, 0.08, 0.05]
    )

    base_item_price = np.random.uniform(
        80,
        350,
        size=num_orders
    )

    order_amount = (
        base_item_price * num_items
    )

    order_amount = np.round(
        order_amount,
        2
    )

    weather = np.random.choice(
        [
            "Clear",
            "Cloudy",
            "Rain",
            "Storm"
        ],
        size=num_orders,
        p=[0.55, 0.25, 0.17, 0.03]
    )

    order_hours = order_timestamps.hour

    peak_hour = (
        ((order_hours >= 11) & (order_hours <= 14)) |
        ((order_hours >= 18) & (order_hours <= 22))
    )

    traffic = []

    for is_peak in peak_hour:

        if is_peak:
            traffic.append(
                np.random.choice(
                    ["Low", "Medium", "High"],
                    p=[0.10, 0.45, 0.45]
                )
            )

        else:
            traffic.append(
                np.random.choice(
                    ["Low", "Medium", "High"],
                    p=[0.55, 0.35, 0.10]
                )
            )

    order_status = np.random.choice(
        ["Completed", "Cancelled"],
        size=num_orders,
        p=[0.92, 0.08]
    )

    orders = pd.DataFrame({
        "order_id": order_ids,
        "customer_id": selected_customers,
        "restaurant_id": selected_restaurants,
        "driver_id": selected_drivers,
        "order_timestamp": order_timestamps,
        "order_amount": order_amount,
        "num_items": num_items,
        "order_status": order_status,
        "city": order_cities,
        "weather": weather,
        "traffic_condition": traffic
    })

    return orders


def generate_deliveries(
    orders,
    restaurants,
    drivers,
    num_deliveries=NUM_DELIVERIES
):
    """Generate synthetic delivery operational data."""

    selected_orders = orders.sample(
        n=num_deliveries,
        replace=False,
        random_state=RANDOM_SEED
    ).copy()

    delivery_ids = [
        f"DLY{str(i).zfill(6)}"
        for i in range(1, num_deliveries + 1)
    ]

    restaurant_prep_lookup = restaurants.set_index(
        "restaurant_id"
    )["avg_prep_time"]

    preparation_time = selected_orders[
        "restaurant_id"
    ].map(
        restaurant_prep_lookup
    ).to_numpy()

    # Add small operational variation
    preparation_time = (
        preparation_time
        + np.random.randint(
            -5,
            6,
            size=num_deliveries
        )
    )

    preparation_time = np.maximum(
        preparation_time,
        5
    )

    delivery_distance = np.random.gamma(
        shape=2.2,
        scale=2.0,
        size=num_deliveries
    )

    delivery_distance = np.clip(
        delivery_distance,
        0.5,
        20
    )

    delivery_distance = np.round(
        delivery_distance,
        2
    )

    traffic_effect = (
        selected_orders["traffic_condition"]
        .map({
            "Low": 0,
            "Medium": 8,
            "High": 18
        })
        .to_numpy()
    )

    weather_effect = (
        selected_orders["weather"]
        .map({
            "Clear": 0,
            "Cloudy": 2,
            "Rain": 8,
            "Storm": 15
        })
        .to_numpy()
    )

    order_hours = (
        selected_orders["order_timestamp"]
        .dt.hour
    )

    peak_hour = (
        (
            (order_hours >= 11) &
            (order_hours <= 14)
        )
        |
        (
            (order_hours >= 18) &
            (order_hours <= 22)
        )
    )

    peak_effect = np.where(
        peak_hour,
        6,
        0
    )

    driver_vehicle_lookup = drivers.set_index(
        "driver_id"
    )["vehicle_type"]

    vehicle_types = selected_orders[
        "driver_id"
    ].map(
        driver_vehicle_lookup
    )

    vehicle_effect = vehicle_types.map({
        "Bike": 3,
        "Scooter": 1,
        "Car": 0
    }).to_numpy()
    

    distance_effect = (
        delivery_distance * 3.5
    )

    random_noise = np.random.normal(
        loc=0,
        scale=4,
        size=num_deliveries
    )

    delivery_time = (
        preparation_time
        + distance_effect
        + traffic_effect
        + weather_effect
        + peak_effect
        + vehicle_effect
        + random_noise
    )

    delivery_time = np.maximum(
        delivery_time,
        10
    )

    delivery_time = np.round(
        delivery_time,
        1
    )

    tip_amount = (
        selected_orders["order_amount"].to_numpy()
        * np.random.uniform(
            0.03,
            0.15,
            size=num_deliveries
        )
    )

    tip_amount = np.round(
        tip_amount,
        2
    )

    delivery_status = np.where(
        selected_orders["order_status"] == "Completed",
        "Delivered",
        "Failed"
    )

    # Cancelled orders do not have an actual delivery
    delivery_time = np.where(
        delivery_status == "Failed",
        0,
        delivery_time
    )

    tip_amount = np.where(
        delivery_status == "Failed",
        0,
        tip_amount
    )


    deliveries = pd.DataFrame({
        "delivery_id": delivery_ids,
        "order_id": selected_orders["order_id"].to_numpy(),
        "delivery_distance_km": delivery_distance,
        "preparation_time_min": np.round(
            preparation_time,
            1
        ),
        "delivery_time_min": delivery_time,
        "tip_amount": tip_amount,
        "delivery_status": delivery_status
    })

    return deliveries


def generate_reviews(
    orders,
    num_reviews=NUM_REVIEWS
):
    """Generate synthetic customer review data."""

    completed_orders = orders[
        orders["order_status"] == "Completed"
    ].copy()

    # Select reviews from completed orders
    selected_orders = completed_orders.sample(
        n=num_reviews,
        replace=False,
        random_state=RANDOM_SEED
    ).copy()

    review_ids = [
        f"REV{str(i).zfill(6)}"
        for i in range(1, num_reviews + 1)
    ]

    ratings = np.random.choice(
        [1, 2, 3, 4, 5],
        size=num_reviews,
        p=[0.06, 0.09, 0.15, 0.35, 0.35]
    )

    positive_reviews = [
        "The food was delicious and fresh.",
        "Amazing food and excellent quality.",
        "The order arrived quickly and everything was hot.",
        "Fast delivery and great packaging.",
        "Really tasty food with generous portions.",
        "Excellent experience from start to finish.",
        "The food was fresh and flavorful.",
        "Delivery was fast and the food arrived hot.",
        "Great quality and very good service.",
        "Everything was perfect and well packed."
    ]

    negative_reviews = [
        "The delivery was very late.",
        "The food arrived cold.",
        "The order took too long to arrive.",
        "The food was bland and disappointing.",
        "My order was missing an item.",
        "I received the wrong item.",
        "The packaging was damaged.",
        "The food was cold and the delivery was late.",
        "The order was delayed for a long time.",
        "Very poor experience with the delivery."
    ]

    neutral_reviews = [
        "The food was okay.",
        "Average experience.",
        "The food was acceptable.",
        "Nothing special about the order.",
        "The experience was fine.",
        "Food quality was average.",
        "The order was okay overall.",
        "It was an ordinary experience.",
        "The food was decent.",
        "Overall it was satisfactory."
    ]

    sentiments = []

    for rating in ratings:

        if rating >= 4:
            sentiment = np.random.choice(
                ["Positive", "Negative"],
                p=[0.95, 0.05]
            )

        elif rating == 3:
            sentiment = np.random.choice(
                ["Neutral", "Positive", "Negative"],
                p=[0.75, 0.15, 0.10]
            )

        else:
            sentiment = np.random.choice(
                ["Negative", "Positive"],
                p=[0.95, 0.05]
            )

        sentiments.append(sentiment)

    review_text = []

    for sentiment in sentiments:

        if sentiment == "Positive":
            text = np.random.choice(
                positive_reviews
            )

        elif sentiment == "Negative":
            text = np.random.choice(
                negative_reviews
            )

        else:
            text = np.random.choice(
                neutral_reviews
            )

        review_text.append(text)

    order_timestamps = pd.to_datetime(
        selected_orders["order_timestamp"]
    )

    review_delay_hours = np.random.randint(
        1,
        49,
        size=num_reviews
    )

    review_timestamps = (
        order_timestamps
        + pd.to_timedelta(
            review_delay_hours,
            unit="h"
        )
    )

    reviews = pd.DataFrame({
        "review_id": review_ids,
        "order_id": selected_orders["order_id"].to_numpy(),
        "customer_id": selected_orders["customer_id"].to_numpy(),
        "restaurant_id": selected_orders["restaurant_id"].to_numpy(),
        "rating": ratings,
        "review_text": review_text,
        "sentiment": sentiments,
        "review_timestamp": review_timestamps
    })

    return reviews


def generate_payments(
    orders,
    num_payments=NUM_PAYMENTS
):
    """Generate synthetic payment transaction data."""

    selected_orders = orders.copy()

    payment_ids = [
        f"PAY{str(i).zfill(6)}"
        for i in range(1, num_payments + 1)
    ]

    payment_methods = np.random.choice(
        [
            "UPI",
            "Credit Card",
            "Debit Card",
            "Wallet",
            "Cash"
        ],
        size=num_payments,
        p=[0.40, 0.25, 0.15, 0.12, 0.08]
    )

    payment_status = []

    for order_status in selected_orders["order_status"]:

        if order_status == "Completed":

            status = np.random.choice(
                ["Successful", "Refunded"],
                p=[0.97, 0.03]
            )

        else:

            status = np.random.choice(
                ["Failed", "Refunded"],
                p=[0.70, 0.30]
            )

        payment_status.append(status)

    order_amounts = (
        selected_orders["order_amount"]
        .to_numpy()
    )

    amount_paid = order_amounts.copy()

    amount_paid = np.where(
        np.array(payment_status) == "Failed",
        0,
        amount_paid
    )

    amount_paid = np.round(
        amount_paid,
        2
    )

    order_timestamps = pd.to_datetime(
        selected_orders["order_timestamp"]
    )

    payment_delay_minutes = np.random.randint(
        0,
        31,
        size=num_payments
    )

    payment_timestamps = (
        order_timestamps
        + pd.to_timedelta(
            payment_delay_minutes,
            unit="m"
        )
    )

    payments = pd.DataFrame({
        "payment_id": payment_ids,
        "order_id": selected_orders["order_id"].to_numpy(),
        "customer_id": selected_orders["customer_id"].to_numpy(),
        "payment_method": payment_methods,
        "payment_status": payment_status,
        "amount_paid": amount_paid,
        "payment_timestamp": payment_timestamps
    })

    return payments


def validate_customers(customers):
    """Validate the generated customer dataset."""

    print("\nCustomer Validation:")

    print("Shape:", customers.shape)

    print(
        "Duplicate customer IDs:",
        customers["customer_id"].duplicated().sum()
    )

    print(
        "Missing values:",
        customers.isnull().sum().sum()
    )

    print(
        "Age range:",
        customers["age"].min(),
        "to",
        customers["age"].max()
    )

    print(
        "Gender values:",
        customers["gender"].unique()
    )


def validate_restaurants(restaurants):
    """Validate the generated restaurant dataset."""

    print("\nRestaurant Validation:")

    print("Shape:", restaurants.shape)

    print(
        "Duplicate restaurant IDs:",
        restaurants["restaurant_id"].duplicated().sum()
    )

    print(
        "Missing values:",
        restaurants.isnull().sum().sum()
    )

    print(
        "Rating range:",
        restaurants["restaurant_rating"].min(),
        "to",
        restaurants["restaurant_rating"].max()
    )

    print(
        "Preparation time range:",
        restaurants["avg_prep_time"].min(),
        "to",
        restaurants["avg_prep_time"].max()
    )

    print(
        "\nRestaurant Categories:"
    )

    print(
        restaurants["restaurant_category"].value_counts()
    )


def validate_drivers(drivers):
    """Validate the generated driver dataset."""

    print("\nDriver Validation:")

    print("Shape:", drivers.shape)

    print(
        "Duplicate driver IDs:",
        drivers["driver_id"].duplicated().sum()
    )

    print(
        "Missing values:",
        drivers.isnull().sum().sum()
    )

    print(
        "Rating range:",
        drivers["driver_rating"].min(),
        "to",
        drivers["driver_rating"].max()
    )

    print(
        "Experience range:",
        drivers["experience_years"].min(),
        "to",
        drivers["experience_years"].max()
    )

    print("\nVehicle distribution:")
    print(
        drivers["vehicle_type"].value_counts()
    )


def validate_orders(
    orders,
    customers,
    restaurants,
    drivers
):
    """Validate the generated order dataset."""

    print("\nOrder Validation:")

    print("Shape:", orders.shape)

    print(
        "Duplicate order IDs:",
        orders["order_id"].duplicated().sum()
    )

    print(
        "Missing values:",
        orders.isnull().sum().sum()
    )

    print(
        "Order amount range:",
        round(orders["order_amount"].min(), 2),
        "to",
        round(orders["order_amount"].max(), 2)
    )

    print(
        "Items range:",
        orders["num_items"].min(),
        "to",
        orders["num_items"].max()
    )

    print(
        "Order date range:",
        orders["order_timestamp"].min(),
        "to",
        orders["order_timestamp"].max()
    )

    print("\nOrder Status:")
    print(
        orders["order_status"].value_counts()
    )

    print("\nWeather Distribution:")
    print(
        orders["weather"].value_counts()
    )

    print("\nTraffic Distribution:")
    print(
        orders["traffic_condition"].value_counts()
    )

    invalid_customers = (
        ~orders["customer_id"].isin(
            customers["customer_id"]
        )
    ).sum()

    invalid_restaurants = (
        ~orders["restaurant_id"].isin(
            restaurants["restaurant_id"]
        )
    ).sum()

    invalid_drivers = (
        ~orders["driver_id"].isin(
            drivers["driver_id"]
        )
    ).sum()

    print("\nReferential Integrity:")
    print(
        "Invalid customer IDs:",
        invalid_customers
    )
    print(
        "Invalid restaurant IDs:",
        invalid_restaurants
    )
    print(
        "Invalid driver IDs:",
        invalid_drivers
    )


def validate_deliveries(
    deliveries,
    orders
):
    """Validate the generated delivery dataset."""

    print("\nDelivery Validation:")

    print("Shape:", deliveries.shape)

    print(
        "Duplicate delivery IDs:",
        deliveries["delivery_id"].duplicated().sum()
    )

    print(
        "Duplicate order IDs:",
        deliveries["order_id"].duplicated().sum()
    )

    print(
        "Missing values:",
        deliveries.isnull().sum().sum()
    )

    print(
        "Distance range:",
        deliveries["delivery_distance_km"].min(),
        "to",
        deliveries["delivery_distance_km"].max()
    )

    print(
        "Preparation time range:",
        deliveries["preparation_time_min"].min(),
        "to",
        deliveries["preparation_time_min"].max()
    )

    delivered = deliveries[
        deliveries["delivery_status"] == "Delivered"
    ]

    print(
        "Delivered records:",
        len(delivered)
    )

    if not delivered.empty:
        print(
            "Delivery time range:",
            delivered["delivery_time_min"].min(),
            "to",
            delivered["delivery_time_min"].max()
        )

        print(
            "Tip amount range:",
            delivered["tip_amount"].min(),
            "to",
            delivered["tip_amount"].max()
        )

    print("\nDelivery Status:")
    print(
        deliveries["delivery_status"].value_counts()
    )

    invalid_orders = (
        ~deliveries["order_id"].isin(
            orders["order_id"]
        )
    ).sum()

    print(
        "\nInvalid order IDs:",
        invalid_orders
    )


def validate_reviews(
    reviews,
    orders
):
    """Validate the generated review dataset."""

    print("\nReview Validation:")

    print(
        "Shape:",
        reviews.shape
    )

    print(
        "Duplicate review IDs:",
        reviews["review_id"].duplicated().sum()
    )

    print(
        "Duplicate order IDs:",
        reviews["order_id"].duplicated().sum()
    )

    print(
        "Missing values:",
        reviews.isnull().sum().sum()
    )

    print(
        "Rating range:",
        reviews["rating"].min(),
        "to",
        reviews["rating"].max()
    )

    print("\nRating Distribution:")
    print(
        reviews["rating"].value_counts().sort_index()
    )

    print("\nSentiment Distribution:")
    print(
        reviews["sentiment"].value_counts()
    )

    print("\nAverage Rating by Sentiment:")
    print(
        reviews.groupby("sentiment")["rating"]
        .mean()
        .sort_values()
    )

    print("\nReview Date Range:")
    print(
        reviews["review_timestamp"].min(),
        "to",
        reviews["review_timestamp"].max()
    )

    invalid_orders = (
        ~reviews["order_id"].isin(
            orders["order_id"]
        )
    ).sum()

    invalid_customers = (
        ~reviews["customer_id"].isin(
            orders["customer_id"]
        )
    ).sum()

    invalid_restaurants = (
        ~reviews["restaurant_id"].isin(
            orders["restaurant_id"]
        )
    ).sum()

    print("\nReferential Integrity:")
    print(
        "Invalid order IDs:",
        invalid_orders
    )

    print(
        "Invalid customer IDs:",
        invalid_customers
    )

    print(
        "Invalid restaurant IDs:",
        invalid_restaurants
    )


def validate_payments(
    payments,
    orders
):
    """Validate the generated payment dataset."""

    print("\nPayment Validation:")

    print(
        "Shape:",
        payments.shape
    )

    print(
        "Duplicate payment IDs:",
        payments["payment_id"].duplicated().sum()
    )

    print(
        "Duplicate order IDs:",
        payments["order_id"].duplicated().sum()
    )

    print(
        "Missing values:",
        payments.isnull().sum().sum()
    )

    print(
        "Amount range:",
        payments["amount_paid"].min(),
        "to",
        payments["amount_paid"].max()
    )

    print("\nPayment Method Distribution:")
    print(
        payments["payment_method"].value_counts()
    )

    print("\nPayment Status Distribution:")
    print(
        payments["payment_status"].value_counts()
    )

    print("\nAverage Amount by Payment Status:")
    print(
        payments.groupby("payment_status")["amount_paid"]
        .mean()
    )

    failed_payments_with_amount = (
        payments[
            payments["payment_status"] == "Failed"
        ]["amount_paid"] > 0
    ).sum()

    print(
        "\nFailed payments with amount > 0:",
        failed_payments_with_amount
    )

    invalid_orders = (
        ~payments["order_id"].isin(
            orders["order_id"]
        )
    ).sum()

    invalid_customers = (
        ~payments["customer_id"].isin(
            orders["customer_id"]
        )
    ).sum()

    print("\nReferential Integrity:")

    print(
        "Invalid order IDs:",
        invalid_orders
    )

    print(
        "Invalid customer IDs:",
        invalid_customers
    )


def save_customers(customers):
    """Save customer dataset to the raw data directory."""

    output_path = Path("data/raw/customers.csv")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    customers.to_csv(
        output_path,
        index=False
    )

    print(f"Customer dataset saved to: {output_path}")
    print(f"Shape: {customers.shape}")


def save_restaurants(restaurants):
    """Save restaurant dataset to the raw data directory."""

    output_path = Path("data/raw/restaurants.csv")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    restaurants.to_csv(
        output_path,
        index=False
    )

    print(
        f"Restaurant dataset saved to: {output_path}"
    )

    print(
        f"Shape: {restaurants.shape}"
    )


def save_drivers(drivers):
    """Save driver dataset to the raw data directory."""

    output_path = Path("data/raw/drivers.csv")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    drivers.to_csv(
        output_path,
        index=False
    )

    print(
        f"Driver dataset saved to: {output_path}"
    )

    print(
        f"Shape: {drivers.shape}"
    )


def save_orders(orders):
    """Save order dataset to the raw data directory."""

    output_path = Path("data/raw/orders.csv")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    orders.to_csv(
        output_path,
        index=False
    )

    print(
        f"Order dataset saved to: {output_path}"
    )

    print(
        f"Shape: {orders.shape}"
    )


def save_deliveries(deliveries):
    """Save delivery dataset to the raw data directory."""

    output_path = Path("data/raw/deliveries.csv")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    deliveries.to_csv(
        output_path,
        index=False
    )

    print(
        f"Delivery dataset saved to: {output_path}"
    )

    print(
        f"Shape: {deliveries.shape}"
    )


def save_reviews(reviews):
    """Save review dataset to the raw data directory."""

    output_path = Path(
        "data/raw/reviews.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    reviews.to_csv(
        output_path,
        index=False
    )

    print(
        f"Review dataset saved to: {output_path}"
    )

    print(
        f"Shape: {reviews.shape}"
    )


def save_payments(payments):
    """Save payment dataset to the raw data directory."""

    output_path = Path(
        "data/raw/payments.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    payments.to_csv(
        output_path,
        index=False
    )

    print(
        f"Payment dataset saved to: {output_path}"
    )

    print(
        f"Shape: {payments.shape}"
    )


if __name__ == "__main__":

    # Customers
    customers = generate_customers()

    print("\nCustomer Dataset Preview:")
    print(customers.head())

    validate_customers(customers)

    save_customers(customers)

    # Restaurants
    restaurants = generate_restaurants()

    print("\nRestaurant Dataset Preview:")
    print(restaurants.head())

    validate_restaurants(restaurants)

    save_restaurants(restaurants)

    # Drivers
    drivers = generate_drivers()

    print("\nDriver Dataset Preview:")
    print(drivers.head())

    validate_drivers(drivers)

    save_drivers(drivers)

    # Orders
    orders = generate_orders(
        customers,
        restaurants,
        drivers
    )

    print("\nOrder Dataset Preview:")
    print(orders.head())

    validate_orders(
        orders,
        customers,
        restaurants,
        drivers
    )

    save_orders(orders)

    # Deliveries
    deliveries = generate_deliveries(
        orders,
        restaurants,
        drivers
    )

    print("\nDelivery Dataset Preview:")
    print(deliveries.head())

    validate_deliveries(
        deliveries,
        orders
    )

    save_deliveries(deliveries)

    # Reviews
    reviews = generate_reviews(
        orders
    )

    print("\nReview Dataset Preview:")
    print(reviews.head())

    validate_reviews(
        reviews,
        orders
    )

    save_reviews(
        reviews
    )

    # Payments
    payments = generate_payments(
        orders
    )

    print("\nPayment Dataset Preview:")
    print(payments.head())

    validate_payments(
        payments,
        orders
    )

    save_payments(
        payments
    )