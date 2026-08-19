from pathlib import Path
import string

import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "reviews.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"

CLEANED_PATH = PROCESSED_DIR / "reviews_cleaned.csv"
TFIDF_PATH = PROCESSED_DIR / "tfidf_features.csv"
THEMES_PATH = PROCESSED_DIR / "negative_review_themes.csv"
PREDICTIONS_PATH = (
    PREDICTIONS_DIR / "sentiment_predictions.csv"
)
CONFUSION_MATRIX_PATH = (
    FIGURES_DIR / "sentiment_confusion_matrix.png"
)


REQUIRED_COLUMNS = [
    "review_id",
    "order_id",
    "customer_id",
    "restaurant_id",
    "rating",
    "review_text",
    "sentiment",
    "review_timestamp",
]


# ============================================================
# NLP RESOURCES
# ============================================================

STOP_WORDS = set(stopwords.words("english"))

# Keep words that can change sentiment meaning.
STOP_WORDS -= {"not", "no", "nor", "never"}

LEMMATIZER = WordNetLemmatizer()


# ============================================================
# DATA LOADING AND VALIDATION
# ============================================================

def load_reviews():
    """Load and validate the expected review dataset schema."""

    df = pd.read_csv(RAW_DATA_PATH)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return df


def validate_reviews(df):
    """Check basic data quality before NLP processing."""

    print("\n" + "=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)

    print("\nDataset shape:")
    print(df.shape)

    print("\nMissing values:")
    print(df[REQUIRED_COLUMNS].isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts())

    print("\nUnique sentiment values:")
    print(df["sentiment"].unique())

    empty_reviews = (
        df["review_text"]
        .fillna("")
        .str.strip()
        .eq("")
        .sum()
    )

    print("\nEmpty reviews:")
    print(empty_reviews)

    if empty_reviews:
        raise ValueError("Empty review texts detected.")


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def clean_text(text):
    """
    Prepare review text for traditional NLP models.

    Steps:
    - Lowercase
    - Replace punctuation with spaces
    - Tokenize
    - Remove stop words
    - Preserve negations
    - Lemmatize
    """

    text = text.lower()

    # Replace punctuation with spaces so words are not joined.
    text = text.translate(
        str.maketrans(
            string.punctuation,
            " " * len(string.punctuation),
        )
    )

    tokens = word_tokenize(text)

    cleaned_tokens = []

    for token in tokens:

        if not token.isalpha():
            continue

        if token in STOP_WORDS:
            continue

        cleaned_tokens.append(
            LEMMATIZER.lemmatize(token)
        )

    return " ".join(cleaned_tokens)


def preprocess_reviews(df):
    """Create a cleaned review column."""

    df = df.copy()

    df["cleaned_review"] = (
        df["review_text"].apply(clean_text)
    )

    empty_cleaned = (
        df["cleaned_review"]
        .str.strip()
        .eq("")
        .sum()
    )

    print("\n" + "=" * 60)
    print("TEXT PREPROCESSING")
    print("=" * 60)

    print(
        "\nEmpty cleaned reviews:",
        empty_cleaned,
    )

    print("\nOriginal vs cleaned reviews:")

    print(
        df[
            ["review_text", "cleaned_review", "sentiment"]
        ]
        .head(15)
        .to_string(index=False)
    )

    return df


def save_cleaned_reviews(df):
    """Save the cleaned review dataset."""

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        CLEANED_PATH,
        index=False,
    )

    print("\nCleaned dataset saved to:")
    print(CLEANED_PATH)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_data(df):
    """Create a stratified train/test split."""

    X = df["cleaned_review"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\n" + "=" * 60)
    print("TRAIN / TEST SPLIT")
    print("=" * 60)

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    print("\nTraining sentiment distribution:")
    print(y_train.value_counts(normalize=True))

    print("\nTesting sentiment distribution:")
    print(y_test.value_counts(normalize=True))

    return X_train, X_test, y_train, y_test


# ============================================================
# TF-IDF
# ============================================================

def create_tfidf(X_train, X_test):
    """
    Convert text into TF-IDF features.

    Includes:
    - Unigrams
    - Bigrams
    """

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=10000,
        sublinear_tf=True,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    feature_names = vectorizer.get_feature_names_out()

    print("\n" + "=" * 60)
    print("TF-IDF")
    print("=" * 60)

    print(
        "\nTraining TF-IDF shape:",
        X_train_tfidf.shape,
    )

    print(
        "Testing TF-IDF shape:",
        X_test_tfidf.shape,
    )

    print(
        "Vocabulary size:",
        len(feature_names),
    )

    return (
        vectorizer,
        X_train_tfidf,
        X_test_tfidf,
        feature_names,
    )


def analyze_tfidf(
    X_train_tfidf,
    feature_names,
):
    """Identify terms with the highest average TF-IDF."""

    mean_scores = X_train_tfidf.mean(axis=0).A1

    tfidf_df = pd.DataFrame(
        {
            "term": feature_names,
            "mean_tfidf": mean_scores,
        }
    ).sort_values(
        "mean_tfidf",
        ascending=False,
    )

    print("\n" + "=" * 60)
    print("TOP TF-IDF WORDS / PHRASES")
    print("=" * 60)

    print(
        tfidf_df.head(30).to_string(index=False)
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tfidf_df.head(100).to_csv(
        TFIDF_PATH,
        index=False,
    )

    return tfidf_df


# ============================================================
# SVM
# ============================================================

def train_svm(X_train_tfidf, y_train):
    """Train a Linear SVM sentiment classifier."""

    model = LinearSVC(
        class_weight="balanced",
        random_state=42,
    )

    model.fit(
        X_train_tfidf,
        y_train,
    )

    print("\n" + "=" * 60)
    print("SVM MODEL")
    print("=" * 60)

    print("\nLinear SVM training completed.")

    return model


def evaluate_model(
    model,
    X_test_tfidf,
    y_test,
):
    """Evaluate the SVM using standard classification metrics."""

    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )
    )

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=model.classes_,
    )

    print("\nConfusion Matrix:")
    print(cm)

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=model.classes_,
    )

    display.plot(
        xticks_rotation="vertical"
    )

    plt.title("SVM Sentiment Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return predictions


def analyze_class_terms(
    model,
    vectorizer,
):
    """Display terms most strongly associated with each class."""

    feature_names = (
        vectorizer.get_feature_names_out()
    )

    print("\n" + "=" * 60)
    print("SENTIMENT-SPECIFIC TERMS")
    print("=" * 60)

    for class_index, class_name in enumerate(
        model.classes_
    ):

        coefficients = model.coef_[class_index]

        top_indices = (
            coefficients.argsort()[-20:][::-1]
        )

        top_terms = feature_names[top_indices]

        print(
            f"\nTop terms associated with {class_name}:"
        )

        print(
            ", ".join(top_terms)
        )


# ============================================================
# VADER
# ============================================================

def run_vader(df):
    """Apply VADER sentiment analysis to original reviews."""

    try:
        analyzer = SentimentIntensityAnalyzer()

    except LookupError:
        import nltk

        nltk.download("vader_lexicon")

        analyzer = SentimentIntensityAnalyzer()

    scores = df["review_text"].apply(
        analyzer.polarity_scores
    )

    df["vader_compound"] = scores.apply(
        lambda x: x["compound"]
    )

    def classify(score):

        if score >= 0.05:
            return "Positive"

        if score <= -0.05:
            return "Negative"

        return "Neutral"

    df["vader_sentiment"] = (
        df["vader_compound"].apply(classify)
    )

    print("\n" + "=" * 60)
    print("VADER SENTIMENT")
    print("=" * 60)

    print("\nVADER sentiment distribution:")

    print(
        df["vader_sentiment"].value_counts()
    )

    return df


# ============================================================
# MODEL COMPARISON
# ============================================================

def compare_models(
    df,
    test_indices,
    ml_predictions,
):
    """Compare SVM predictions with VADER on the test set."""

    comparison = df.loc[
        test_indices,
        [
            "review_id",
            "review_text",
            "sentiment",
            "vader_sentiment",
            "vader_compound",
        ],
    ].copy()

    comparison["ml_sentiment"] = ml_predictions

    ml_accuracy = (
        comparison["ml_sentiment"]
        == comparison["sentiment"]
    ).mean()

    vader_accuracy = (
        comparison["vader_sentiment"]
        == comparison["sentiment"]
    ).mean()

    agreement = (
        comparison["ml_sentiment"]
        == comparison["vader_sentiment"]
    ).mean()

    print("\n" + "=" * 60)
    print("ML VS VADER")
    print("=" * 60)

    print(
        f"\nML accuracy on test set: {ml_accuracy:.4f}"
    )

    print(
        f"VADER accuracy on test set: {vader_accuracy:.4f}"
    )

    print(
        f"\nAgreement between ML and VADER: {agreement:.4f}"
    )


# ============================================================
# COMPLAINT / THEME ANALYSIS
# ============================================================

def analyze_complaints(df):
    """
    Quantify recurring complaint themes in negative reviews.

    Themes are based on patterns observed in the dataset.
    A review may match multiple themes.
    """

    negative = df[
        df["sentiment"] == "Negative"
    ].copy()

    total_negative = len(negative)

    patterns = {
        "Delivery Delays": [
            r"\blate\b",
            r"\bdelayed\b",
            r"\btook long\b",
            r"\blong time\b",
            r"\bdelivery late\b",
            r"\blong arrive\b",
        ],
        "Food Quality / Temperature": [
            r"\bcold\b",
            r"\bbland\b",
            r"\bdisappointing\b",
            r"\barrived cold\b",
        ],
        "Wrong / Missing Items": [
            r"\bmissing\b",
            r"\bwrong\b",
            r"\bmissing item\b",
            r"\bwrong item\b",
            r"\border missing\b",
            r"\breceived wrong\b",
        ],
        "Packaging Damage": [
            r"\bdamaged\b",
            r"\bpackaging damaged\b",
        ],
    }

    results = []

    for theme, theme_patterns in patterns.items():

        pattern = "|".join(theme_patterns)

        matches = negative["cleaned_review"].str.contains(
            pattern,
            regex=True,
            na=False,
        )

        count = matches.sum()

        results.append(
            {
                "theme": theme,
                "negative_reviews": count,
                "percentage_of_negative_reviews": (
                    count / total_negative * 100
                ),
            }
        )

    theme_df = pd.DataFrame(results).sort_values(
        "negative_reviews",
        ascending=False,
    )

    print("\n" + "=" * 60)
    print("NEGATIVE REVIEW / COMPLAINT ANALYSIS")
    print("=" * 60)

    print(
        f"\nTotal negative reviews: {total_negative}"
    )

    print("\nCustomer dissatisfaction themes:")

    print(
        theme_df.to_string(
            index=False,
            formatters={
                "percentage_of_negative_reviews":
                    "{:.2f}%".format
            },
        )
    )

    print(
        "\nNote: A single review can belong to "
        "multiple themes."
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    theme_df.to_csv(
        THEMES_PATH,
        index=False,
    )

    print("\nTheme analysis saved to:")
    print(THEMES_PATH)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(
    df,
    test_indices,
    ml_predictions,
):
    """Save model predictions for the test observations."""

    predictions = df.loc[
        test_indices
    ].copy()

    predictions["ml_sentiment"] = ml_predictions

    predictions["ml_correct"] = (
        predictions["ml_sentiment"]
        == predictions["sentiment"]
    )

    predictions["vader_correct"] = (
        predictions["vader_sentiment"]
        == predictions["sentiment"]
    )

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    print("\nPredictions saved to:")
    print(PREDICTIONS_PATH)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("UBER EATS CUSTOMER FEEDBACK NLP PIPELINE")
    print("=" * 60)

    # Load and validate
    df = load_reviews()
    validate_reviews(df)

    # Preprocessing
    df = preprocess_reviews(df)
    save_cleaned_reviews(df)

    # Train/test split
    X_train, X_test, y_train, y_test = split_data(df)
    test_indices = X_test.index

    # TF-IDF
    (
        vectorizer,
        X_train_tfidf,
        X_test_tfidf,
        feature_names,
    ) = create_tfidf(
        X_train,
        X_test,
    )

    analyze_tfidf(
        X_train_tfidf,
        feature_names,
    )

    # SVM
    model = train_svm(
        X_train_tfidf,
        y_train,
    )

    predictions = evaluate_model(
        model,
        X_test_tfidf,
        y_test,
    )

    analyze_class_terms(
        model,
        vectorizer,
    )

    # VADER
    df = run_vader(df)

    compare_models(
        df,
        test_indices,
        predictions,
    )

    # Complaint analysis
    analyze_complaints(df)

    # Save predictions
    save_predictions(
        df,
        test_indices,
        predictions,
    )

    print("\n" + "=" * 60)
    print("NLP PIPELINE COMPLETED")
    print("=" * 60)

    print("\nOutputs:")
    print(CLEANED_PATH)
    print(TFIDF_PATH)
    print(THEMES_PATH)
    print(PREDICTIONS_PATH)
    print(CONFUSION_MATRIX_PATH)


if __name__ == "__main__":
    main()