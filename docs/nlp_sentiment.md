# Customer Feedback Analysis Using NLP

## 1. Overview

Natural Language Processing (NLP) is used in this project to analyze Uber Eats customer reviews and convert unstructured feedback into actionable customer and operational insights.

The NLP pipeline includes:

* Text preprocessing
* Train/test splitting
* TF-IDF feature extraction
* Linear SVM sentiment classification
* Model evaluation
* VADER sentiment analysis
* Negative review analysis
* Complaint theme identification
* Business interpretation

The overall workflow is:

**Raw Reviews → Preprocessing → Train/Test Split → TF-IDF → Linear SVM → Evaluation → VADER → Complaint Analysis → Business Insights**

---

# 2. Objective

The objective of the NLP module is to:

1. Clean and standardize customer review text.
2. Convert text into numerical features using TF-IDF.
3. Classify reviews as Positive, Neutral, or Negative.
4. Evaluate the sentiment classification model.
5. Compare supervised ML with rule-based sentiment analysis.
6. Identify major causes of customer dissatisfaction.
7. Translate customer feedback into operational recommendations.

---

# 3. Dataset

The dataset contains **30,000 customer reviews** and 8 columns.

| Column             | Description                    |
| ------------------ | ------------------------------ |
| `review_id`        | Unique review identifier       |
| `order_id`         | Associated order identifier    |
| `customer_id`      | Customer identifier            |
| `restaurant_id`    | Restaurant identifier          |
| `rating`           | Customer rating                |
| `review_text`      | Written customer feedback      |
| `sentiment`        | Positive, Neutral, or Negative |
| `review_timestamp` | Review creation timestamp      |

## 3.1 Dataset Validation

| Validation        | Result |
| ----------------- | -----: |
| Total records     | 30,000 |
| Missing values    |      0 |
| Duplicate rows    |      0 |
| Empty reviews     |      0 |
| Sentiment classes |      3 |

## 3.2 Sentiment Distribution

| Sentiment |    Reviews | Percentage |
| --------- | ---------: | ---------: |
| Positive  |     20,858 |     69.53% |
| Negative  |      5,700 |     19.00% |
| Neutral   |      3,442 |     11.47% |
| **Total** | **30,000** |   **100%** |

The dataset is imbalanced toward Positive reviews. Therefore, model evaluation considers **precision, recall, and F1-score** in addition to accuracy.

---

# 4. Text Preprocessing

Raw customer reviews cannot be directly processed by traditional machine learning algorithms. The text was therefore cleaned and standardized before feature extraction.

The preprocessing workflow is:

**Raw Text → Lowercasing → Punctuation Removal → Tokenization → Stop-word Removal → Negation Preservation → Lemmatization → Cleaned Text**

### Preprocessing Steps

* **Lowercasing:** Standardizes words such as `Food` and `food`.
* **Punctuation removal:** Reduces unnecessary vocabulary variation.
* **Tokenization:** Splits reviews into individual tokens.
* **Stop-word removal:** Removes common low-information words.
* **Negation preservation:** Retains important words such as `not`, `no`, `nor`, and `never`.
* **Lemmatization:** Converts words into consistent base forms.

For example:

> `The delivery was very late.`

becomes approximately:

> `delivery late`

The original `review_text` is preserved, while the cleaned text is stored in:

`cleaned_review`

### Preprocessing Validation

* Original reviews: **30,000**
* Empty cleaned reviews: **0**

The cleaned dataset was saved to:

`data/processed/reviews_cleaned.csv`

---

# 5. Train/Test Split

The dataset was divided into:

* **80% training data**
* **20% testing data**

| Dataset  | Reviews |
| -------- | ------: |
| Training |  24,000 |
| Testing  |   6,000 |

The split was **stratified by sentiment** so that the Positive, Negative, and Neutral proportions remained approximately consistent across both datasets.

The test set was kept unseen during model training and was used only for final evaluation.

---

# 6. TF-IDF Feature Extraction

The cleaned reviews were converted into numerical features using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

TF-IDF assigns importance to terms based on:

* Their frequency within a review.
* Their frequency across the complete collection of reviews.

The basic representation is:

**TF-IDF = Term Frequency × Inverse Document Frequency**

The resulting matrix contains:

* Rows representing reviews.
* Columns representing vocabulary terms.
* Values representing term importance.

## 6.1 Unigrams and Bigrams

The vectorizer captures both individual words and short phrases.

### Unigrams

* `food`
* `delivery`
* `fresh`
* `quality`

### Bigrams

* `fast delivery`
* `delivery late`
* `packaging damaged`
* `wrong item`

Bigrams provide additional context that individual words may not capture.

## 6.2 TF-IDF Configuration

| Parameter                  | Configuration |
| -------------------------- | ------------- |
| N-gram range               | 1–2           |
| Minimum document frequency | 2             |
| Maximum document frequency | 95%           |
| Maximum features           | 10,000        |
| Sublinear TF scaling       | Enabled       |

The vectorizer was fitted **only on the training data** and then used to transform both training and testing data.

This prevents **data leakage** from the test set.

## 6.3 TF-IDF Output

| Dataset         |        Shape |
| --------------- | -----------: |
| Training TF-IDF | 24,000 × 116 |
| Testing TF-IDF  |  6,000 × 116 |
| Vocabulary size |          116 |

The relatively small vocabulary is consistent with the synthetic dataset, which contains repetitive and controlled review patterns.

Important terms included:

* `food`
* `delivery`
* `fresh`
* `quality`
* `excellent`
* `fast`
* `hot`
* `great`

Important phrases included:

* `fast delivery`
* `great packaging`
* `arrived hot`
* `excellent quality`

A high TF-IDF value does **not automatically indicate sentiment**. Sentiment-specific interpretation is performed using the trained classifier.

The TF-IDF features were saved to:

`data/processed/tfidf_features.csv`

---

# 7. Sentiment Classification Using Linear SVM

The TF-IDF features were used as input to a **Linear Support Vector Machine (Linear SVM)** classifier.

The target classes are:

* Positive
* Negative
* Neutral

## 7.1 Why Linear SVM?

Linear SVM is suitable for this task because:

* TF-IDF produces high-dimensional sparse features.
* Linear SVM performs efficiently on sparse text data.
* It supports multi-class classification.
* Its coefficients can be inspected to identify sentiment-specific terms.

## 7.2 Model Training

The model uses:

* **Input:** TF-IDF training features
* **Target:** Sentiment labels
* **Training samples:** 24,000
* **Classification:** Multi-class

The model was trained only on the training dataset.

Because the sentiment classes are imbalanced, **balanced class weighting** was used so that the majority Positive class does not disproportionately influence training.

---

# 8. Model Evaluation

The Linear SVM was evaluated on the **6,000 unseen test reviews**.

The evaluation metrics were:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

## 8.1 Results

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 1.0000 |
| Precision | 1.0000 |
| Recall    | 1.0000 |
| F1-score  | 1.0000 |

### Classification Report

| Sentiment | Precision | Recall | F1-score | Support |
| --------- | --------: | -----: | -------: | ------: |
| Negative  |      1.00 |   1.00 |     1.00 |   1,140 |
| Neutral   |      1.00 |   1.00 |     1.00 |     688 |
| Positive  |      1.00 |   1.00 |     1.00 |   4,172 |

The model correctly classified all **6,000 test reviews**, resulting in **0 misclassified reviews**.

The confusion matrix was saved to:

`outputs/figures/sentiment_confusion_matrix.png`

## 8.2 Interpreting the Perfect Accuracy

The **100% test accuracy should not be interpreted as expected real-world production performance**.

The dataset is synthetic and contains repetitive, predictable language patterns.

Examples include:

* `late`, `cold`, `damaged`, `missing` → Negative
* `excellent`, `fresh`, `great`, `delicious` → Positive
* `average`, `acceptable`, `okay`, `ordinary` → Neutral

These patterns make the classification problem considerably easier than real-world customer feedback.

Therefore, the result demonstrates that the NLP pipeline performs correctly on this dataset, but it is **not a realistic estimate of production performance**.

---

# 9. Sentiment-Specific Terms

Linear SVM coefficients were examined to identify terms strongly associated with each sentiment.

## 9.1 Negative Sentiment

Important terms included:

* `cold`
* `late`
* `delivery late`
* `packaging damaged`
* `bland`
* `disappointing`
* `missing item`
* `wrong item`

These terms indicate common sources of customer dissatisfaction.

## 9.2 Neutral Sentiment

Important terms included:

* `average`
* `okay`
* `acceptable`
* `decent`
* `ordinary`
* `nothing special`

These generally represent acceptable but unremarkable experiences.

## 9.3 Positive Sentiment

Important terms included:

* `fresh`
* `excellent`
* `fast`
* `hot`
* `great`
* `delicious`
* `fast delivery`
* `great packaging`

These terms are associated with positive food, delivery, and packaging experiences.

---

# 10. VADER Sentiment Analysis

The project also applies **VADER (Valence Aware Dictionary and sEntiment Reasoner)** as a rule-based sentiment analysis approach.

Unlike Linear SVM, VADER does not require training. It uses a predefined sentiment lexicon and linguistic rules to estimate sentiment.

## 10.1 Linear SVM vs VADER

| Linear SVM                       | VADER                              |
| -------------------------------- | ---------------------------------- |
| Supervised ML                    | Rule-based                         |
| Uses labeled training data       | Requires no training               |
| Uses TF-IDF features             | Uses predefined lexicon            |
| Learns dataset-specific patterns | General-purpose sentiment analysis |

## 10.2 VADER Scoring

VADER produces positive, negative, neutral, and compound scores.

The compound score ranges approximately from:

```text
-1 → strongly negative
 0 → neutral
+1 → strongly positive
```

The project converts it into:

* **Positive:** `compound >= 0.05`
* **Negative:** `compound <= -0.05`
* **Neutral:** `-0.05 < compound < 0.05`

## 10.3 VADER Results

| Sentiment | Reviews |
| --------- | ------: |
| Positive  |  18,323 |
| Neutral   |   7,960 |
| Negative  |   3,717 |

The VADER distribution differs from the original labeled distribution because VADER uses its own general-purpose sentiment rules.

---

# 11. ML vs VADER Comparison

Both approaches were evaluated against the original labels on the same **6,000 test reviews**.

| Model      | Test Accuracy |
| ---------- | ------------: |
| Linear SVM |        1.0000 |
| VADER      |        0.7168 |

VADER achieved an accuracy of **71.68%**, while Linear SVM achieved **100%** on this synthetic dataset.

The agreement between the two approaches was also **71.68%**, meaning approximately 71.68% of test reviews received the same sentiment prediction from both methods.

Linear SVM performs better because it learns directly from the project's labeled data and captures dataset-specific expressions such as:

* `delivery late`
* `packaging damaged`
* `wrong item`
* `excellent quality`
* `fast delivery`

VADER relies on a general-purpose lexicon and does not learn from this specific dataset.

Therefore, VADER is more suitable as a **baseline or supplementary approach**, while Linear SVM is the stronger classifier for this dataset.

---

# 12. Negative Review Analysis

Sentiment classification identifies negative reviews, but it does not explain **why customers are dissatisfied**.

Therefore, the **5,700 negative reviews** were analyzed to identify recurring complaint patterns.

Common terms included:

* `delivery`
* `food`
* `late`
* `cold`
* `wrong`
* `missing`
* `damaged`
* `packaging`
* `bland`
* `delayed`

This analysis moves beyond sentiment classification by connecting negative feedback to specific operational issues.

---

# 13. Customer Dissatisfaction Themes

Negative reviews were grouped into four major operational themes.

## 13.1 Theme Results

| Theme                      | Negative Reviews | % of Negative Reviews |
| -------------------------- | ---------------: | --------------------: |
| Delivery Delays            |            2,285 |                40.09% |
| Food Quality / Temperature |            1,716 |                30.11% |
| Wrong / Missing Items      |            1,094 |                19.19% |
| Packaging Damage           |              551 |                 9.67% |

> **Note:** A review can belong to multiple themes, so the percentages do not necessarily sum to 100%.

## 13.2 Delivery Delays

**2,285 negative reviews** were associated with delivery-delay patterns.

Common terms include:

* `late`
* `delayed`
* `long time`
* `delivery late`

### Business Implication

Potential areas for investigation include:

* Peak-hour capacity
* Rider availability
* Traffic-related delays
* Restaurant preparation delays
* Dispatch efficiency

## 13.3 Food Quality / Temperature

**1,716 negative reviews** contained food-quality or temperature complaints.

Common terms include:

* `cold`
* `bland`
* `disappointing`
* `arrived cold`

### Business Implication

Potential areas include:

* Restaurant preparation quality
* Temperature retention
* Preparation time
* Delivery duration
* Restaurant-level quality consistency

## 13.4 Wrong / Missing Items

**1,094 negative reviews** contained complaints about incorrect or missing items.

Common terms include:

* `missing`
* `wrong`
* `missing item`
* `wrong item`

### Business Implication

Potential issues include:

* Order assembly
* Restaurant verification
* Item availability
* Packing processes
* Order handoff

## 13.5 Packaging Damage

**551 negative reviews** contained packaging complaints.

Common terms include:

* `damaged`
* `packaging damaged`

### Business Implication

Packaging issues may affect:

* Food presentation
* Customer satisfaction
* Leakage or spillage
* Restaurant packaging standards

---

# 14. Business Insights and Recommendations

## Insight 1 — Delivery Delays Are the Largest Complaint Category

Delivery delays account for **40.09% of negative reviews**.

### Recommendation

Investigate:

* Peak-hour rider availability
* Dispatch allocation
* Restaurant preparation delays
* High-delay geographic zones
* Delivery capacity constraints

## Insight 2 — Food Quality and Temperature Are Major Dissatisfaction Drivers

Food quality and temperature complaints account for **30.11% of negative reviews**.

### Recommendation

Monitor:

* Restaurant preparation time
* Delivery duration
* Temperature-sensitive food categories
* Packaging quality
* Restaurant-specific complaint rates

## Insight 3 — Order Accuracy Is an Important Operational Issue

Wrong or missing items account for **19.19% of negative reviews**.

### Recommendation

Strengthen:

* Restaurant order verification
* Packing checks
* Item availability monitoring
* Handoff procedures

Restaurants with unusually high missing-item or wrong-item complaint rates should be investigated.

## Insight 4 — Packaging Contributes to Dissatisfaction

Packaging damage accounts for **9.67% of negative reviews**.

### Recommendation

Repeated packaging complaints can be used to identify restaurants requiring:

* Packaging-quality reviews
* Improved packaging standards
* Better handling procedures
* Category-specific packaging recommendations

---

# 15. Limitations

## 15.1 Synthetic Dataset

The dataset contains controlled and repetitive review patterns.

Therefore, the **100% Linear SVM accuracy should not be interpreted as production-level performance**.

## 15.2 Limited Vocabulary

The final TF-IDF vocabulary contains only **116 features**.

Real-world reviews would likely contain:

* Slang
* Abbreviations
* Spelling mistakes
* Emojis
* Mixed languages
* Sarcasm
* Longer and more varied reviews
* Domain-specific terminology

## 15.3 Rule-Based Theme Detection

Complaint themes are identified using predefined text patterns.

This approach is interpretable but may miss complaints expressed using unexpected wording.

More advanced approaches could include:

* Topic modeling
* Embeddings
* Semantic similarity
* Transformer-based models
* BERTopic

## 15.4 VADER Limitations

VADER is a general-purpose sentiment analyzer and may not understand every food-delivery-specific expression.

Therefore, it is best treated as a **comparison or baseline**, rather than automatically as the final production model.

---

# 16. Final NLP Pipeline

```text
Raw Customer Reviews
        ↓
Data Validation
        ↓
Text Preprocessing
        ↓
Train / Test Split
        ↓
TF-IDF Feature Extraction
        ↓
Linear SVM Classification
        ↓
Model Evaluation
        ↓
VADER Sentiment Analysis
        ↓
ML vs VADER Comparison
        ↓
Negative Review Analysis
        ↓
Complaint Theme Identification
        ↓
Business Insights
```

The pipeline combines supervised machine learning, rule-based NLP, and complaint analysis to transform customer feedback into operational intelligence.

---

# 17. Final Findings

| Analysis                      |          Result |
| ----------------------------- | --------------: |
| Total reviews                 |          30,000 |
| Negative reviews              |           5,700 |
| Training reviews              |          24,000 |
| Test reviews                  |           6,000 |
| TF-IDF vocabulary             |             116 |
| Linear SVM accuracy           |            100% |
| VADER accuracy                |          71.68% |
| Largest complaint theme       | Delivery Delays |
| Delivery-delay complaints     |          40.09% |
| Food-quality complaints       |          30.11% |
| Wrong/missing-item complaints |          19.19% |
| Packaging complaints          |           9.67% |

The strongest operational finding is that **delivery delays are the largest identified source of customer dissatisfaction**, followed by food quality/temperature issues, wrong or missing items, and packaging damage.

These findings can be combined with other ML components such as **delivery-time prediction** and **demand forecasting** to support stronger operational decisions.

---

# 18. Output Files

The NLP pipeline generates the following outputs:

| Output                 | Location                                         |
| ---------------------- | ------------------------------------------------ |
| Cleaned Reviews        | `data/processed/reviews_cleaned.csv`             |
| TF-IDF Features        | `data/processed/tfidf_features.csv`              |
| Negative Review Themes | `data/processed/negative_review_themes.csv`      |
| Sentiment Predictions  | `outputs/predictions/sentiment_predictions.csv`  |
| Confusion Matrix       | `outputs/figures/sentiment_confusion_matrix.png` |

### Cleaned Reviews

Contains the original review text and the corresponding cleaned text.

### TF-IDF Features

Contains the extracted TF-IDF terms and their average importance.

### Negative Review Themes

Contains identified customer dissatisfaction themes and their frequencies.

### Sentiment Predictions

Contains sentiment predictions generated by the ML model and VADER.

### Confusion Matrix

Visual representation of Linear SVM classification performance.

---

# 19. Conclusion

The NLP module demonstrates how unstructured customer feedback can be transformed into actionable operational intelligence.

The workflow uses:

* Text preprocessing
* TF-IDF feature extraction
* Linear SVM sentiment classification
* VADER as a rule-based comparison
* Negative-review analysis
* Complaint theme identification

Linear SVM achieved **100% accuracy** on the synthetic test dataset, while VADER achieved **71.68%**.

The most important business finding is that **delivery delays represent the largest identified source of customer dissatisfaction**, accounting for **40.09% of negative reviews**.

The NLP module therefore provides the **customer-feedback intelligence layer** of the overall **Uber Eats Operational & Customer Intelligence System**.
