# Customer Segmentation

## 1. Objective

The objective of this module is to segment customers into meaningful behavioral groups using **unsupervised machine learning**.

The segmentation process uses the customer-level behavioral features created during feature engineering.

The main algorithms used are:

* K-Means Clustering
* DBSCAN
* PCA

The final objective is to identify groups of customers with similar:

* Ordering activity
* Spending behavior
* Ordering frequency
* Weekend ordering behavior
* Late-night ordering behavior
* Rating behavior

The resulting customer segments can be used to support targeted marketing, customer retention, and business decision-making.

---

## 2. Input Dataset

The segmentation module uses the feature-engineered customer dataset:

``` data/processed/customer_features.csv```

The dataset contains one row per customer.

The behavioral features used for clustering are:

* `total_orders`
* `avg_order_value`
* `total_spending`
* `weekend_orders`
* `late_night_orders`
* `ordering_frequency`
* `avg_rating_given`

The `customer_id` column is retained separately for identification but is **not used as a machine learning feature**.

---

## 3. Why Customer IDs Are Excluded

Customer identifiers do not represent customer behavior.

If an identifier were included in a distance-based clustering algorithm, the numerical value of the identifier could create meaningless distances between customers.

Therefore, the segmentation workflow separates:

### Customer Identifier

```
customer_id
```

### Machine Learning Features

``` total_orders```
``` avg_order_value```
``` total_spending```
``` weekend_orders```
``` late_night_orders```
``` ordering_frequency```
``` avg_rating_given```


The customer ID is added back later so that the resulting clusters can be associated with individual customers.

---

# 4. Customer-Level EDA

Before applying clustering algorithms, exploratory analysis is performed on the feature matrix.

The purpose of this step is to understand:

* Feature distributions
* Feature ranges
* Potential outliers
* Relationships between features
* Correlation between behavioral variables

The module generates descriptive statistics and a correlation matrix.

---

## 4.1 Feature Statistics

The customer feature dataset contains:

```
5,000 customers
```

The observed feature ranges are:

| Feature              | Minimum |   Maximum |
| -------------------- | ------: | --------: |
| `total_orders`       |       1 |        24 |
| `avg_order_value`    |  169.72 |  1,354.02 |
| `total_spending`     |  417.01 | 17,403.33 |
| `weekend_orders`     |       0 |        13 |
| `late_night_orders`  |       0 |         8 |
| `ordering_frequency` |    1.00 |      5.50 |
| `avg_rating_given`   |       0 |         5 |

The different scales of these variables are important because the clustering algorithms use distance calculations.

---

## 4.2 Feature Distributions

The module generates a feature distribution figure to visually inspect the distributions of the customer behavioral variables.

### Output

```
outputs/figures/customer_feature_distributions.png
```

This helps identify:

* Skewed distributions
* Concentration of customers
* Unusual values
* Differences in feature ranges

---

## 4.3 Boxplot Analysis

Boxplots are generated for the customer features to inspect potential outliers.

### Output

```
outputs/figures/customer_feature_boxplots.png
```

Outlier inspection is particularly important for distance-based algorithms because extreme observations can influence cluster formation.

---

## 4.4 Correlation Analysis

The correlation matrix shows relationships between customer behavioral features.

Some notable relationships in the current dataset include:

| Feature Pair                            | Correlation |
| --------------------------------------- | ----------: |
| `total_orders` and `total_spending`     |    **0.82** |
| `total_orders` and `ordering_frequency` |    **0.74** |
| `total_orders` and `weekend_orders`     |    **0.61** |
| `avg_order_value` and `total_spending`  |    **0.52** |
| `total_orders` and `late_night_orders`  |    **0.48** |

These relationships indicate that several features capture related aspects of customer engagement and spending.

For example, customers who place more orders are generally likely to have higher total spending.

---

# 5. Feature Scaling

Before clustering, all behavioral features are standardized using **StandardScaler**.

The purpose of scaling is to place all features on a comparable scale.

This is important because K-Means and DBSCAN are distance-based algorithms.

Without scaling, features with larger numerical ranges could dominate the distance calculation.

For example:

```
total_spending
```

has a much larger numerical range than:

```
late_night_orders
```

If the data were not standardized, total spending could have a disproportionately large influence on clustering.

Standardization transforms each feature so that it has approximately:

* Mean = 0
* Standard deviation = 1

The scaled feature matrix is then used for K-Means and DBSCAN.

---

# 6. K-Means Clustering

## 6.1 Objective

K-Means is used to divide customers into a predefined number of behavioral clusters.

The algorithm attempts to group customers such that:

* Customers within the same cluster are similar.
* Customers belonging to different clusters are relatively different.

K-Means is appropriate for this task because the customer representation consists of numerical behavioral features.

---

## 6.2 Selecting the Number of Clusters

The number of clusters is not chosen blindly.

The module evaluates values of:

```
K = 2 to 8
```

Two evaluation measures are calculated:

* Inertia
* Silhouette Score

---

## 6.3 Elbow Method

The **Elbow Method** evaluates the K-Means inertia for different values of K.

Inertia represents the total within-cluster squared distance between observations and their assigned cluster centers.

As K increases, inertia generally decreases.

The objective is to identify a point where increasing K produces diminishing improvement.

### Output

```
outputs/figures/customer_elbow.png
```

---

## 6.4 Silhouette Score

The **Silhouette Score** measures how well-separated the clusters are.

It considers:

* How close a customer is to other customers in its own cluster.
* How far the customer is from customers in other clusters.

The score ranges approximately from:

```
-1 to +1
```

Higher values generally indicate better-defined clusters.

### Output

```
outputs/figures/customer_silhouette.png
```

---

## 6.5 K-Means Evaluation Results

The current results were:

|  K |     Inertia | Silhouette Score |
| -: | ----------: | ---------------: |
|  2 | 24,744.8999 |       **0.2513** |
|  3 | 21,952.3981 |           0.1654 |
|  4 | 19,705.2810 |           0.1640 |
|  5 | 18,096.2414 |           0.1614 |
|  6 | 17,059.2717 |           0.1470 |
|  7 | 16,165.8371 |           0.1463 |
|  8 | 15,475.8920 |           0.1376 |

The highest silhouette score occurs at:

```
K = 2
```

Therefore, the current implementation selects **2 clusters**.

The relatively higher silhouette score for K = 2 also provides stronger separation than the tested alternatives.

---

# 7. Final K-Means Model

The final K-Means model is trained using:

```
Number of clusters = 2
```

The model assigns every customer to one of two clusters.

The resulting cluster distribution is:

| Cluster | Customers |
| ------: | --------: |
|       0 |     2,819 |
|       1 |     2,181 |

This represents:

* **Cluster 0** → approximately 56.4% of customers
* **Cluster 1** → approximately 43.6% of customers

---

# 8. Cluster Profiling

After assigning customers to clusters, the behavioral characteristics of each cluster are calculated.

The profile uses the mean values of the customer features.

The resulting profile is:

| Feature              | Cluster 0 | Cluster 1 |
| -------------------- | --------: | --------: |
| Customer Count       |     2,819 |     2,181 |
| Total Orders         |      7.21 |     11.80 |
| Average Order Value  |    549.17 |    591.85 |
| Total Spending       |  3,909.19 |  6,921.34 |
| Ordering Frequency   |      1.68 |      2.33 |
| Average Rating Given |      3.84 |      3.84 |
| Weekend Orders       |      2.54 |      4.67 |
| Late-Night Orders    |      1.59 |      2.99 |

The profile is used to understand the behavioral differences between the clusters.

---

# 9. Business Interpretation of K-Means Clusters

## Cluster 0 – Occasional / Lower-Engagement Customers

Cluster 0 contains:

```
2,819 customers
```

These customers have relatively:

* Fewer total orders
* Lower total spending
* Lower ordering frequency
* Fewer weekend orders
* Fewer late-night orders
* Slightly lower average order value

The segment is therefore interpreted as:

**Occasional / Lower-Engagement Customers**

### Potential Business Interpretation

These customers may have lower platform engagement and could represent an opportunity for:

* Re-engagement campaigns
* Personalized offers
* Frequency-based promotions
* Recommendations based on previous orders

---

## Cluster 1 – Highly Engaged / High-Value Customers

Cluster 1 contains:

```
2,181 customers
```

These customers have relatively:

* More total orders
* Higher total spending
* Higher ordering frequency
* Higher average order value
* More weekend orders
* More late-night orders

The segment is therefore interpreted as:

**Highly Engaged / High-Value Customers**

### Potential Business Interpretation

These customers demonstrate stronger engagement and monetary contribution.

They may be suitable for:

* Loyalty programs
* Premium customer benefits
* Personalized recommendations
* Retention campaigns
* Cross-selling opportunities

---

# 10. DBSCAN Clustering

## 10.1 Objective

DBSCAN is used as a second clustering approach.

Unlike K-Means, DBSCAN does not require the number of clusters to be specified beforehand.

DBSCAN groups observations based on density and can identify:

* Dense groups
* Sparse regions
* Noise points
* Potential outliers

This makes DBSCAN useful for comparing a centroid-based clustering approach with a density-based approach.

---

# 11. DBSCAN Parameter Evaluation

The module evaluates several `eps` values while keeping:

```
min_samples = 10
```

The tested values are:

```
0.5
0.7
0.8
0.9
1.1
1.3
```

The results were:

| Epsilon | Clusters | Noise Points |
| ------: | -------: | -----------: |
|     0.5 |        0 |        5,000 |
|     0.7 |       10 |        4,020 |
|     0.8 |        3 |        2,693 |
|     0.9 |        1 |        1,569 |
|     1.1 |        1 |          513 |
|     1.3 |        1 |          186 |

The current implementation selects:

```
eps = 0.8
min_samples = 10
```

This produces three DBSCAN clusters and a substantial number of noise points.

---

# 12. DBSCAN Results

Using the selected parameters:

```
eps = 0.8
min_samples = 10
```

DBSCAN identifies:

```
3 clusters
```

and:

```
2,693 noise points
```

The noise percentage is:

```
53.86%
```

A DBSCAN label of `-1` represents a noise observation.

Therefore, more than half of the customers are considered noise under the selected DBSCAN parameters.

---

# 13. K-Means vs DBSCAN

The two clustering methods are compared using:

* Number of clusters
* Silhouette Score
* Number of noise points
* Noise percentage

The current comparison is:

| Method  | Clusters | Silhouette Score | Noise Points | Noise % |
| ------- | -------: | ---------------: | -----------: | ------: |
| K-Means |        2 |       **0.2513** |            0 |   0.00% |
| DBSCAN  |        3 |           0.1293 |        2,693 |  53.86% |

Based on the current results, K-Means provides:

* A higher silhouette score
* More practical cluster coverage
* No noise observations
* Easier business interpretation

Therefore, **K-Means is selected as the primary customer segmentation approach** for the current project.

DBSCAN is retained as a comparison method to demonstrate an alternative clustering technique.

---

# 14. PCA for Cluster Visualization

## 14.1 Objective

Principal Component Analysis (PCA) is used to reduce the seven-dimensional customer feature space to two dimensions.

The purpose is **visualization rather than replacing the original clustering features**.

The PCA transformation is performed on the standardized features.

---

## 14.2 Explained Variance

The current PCA results are:

| Component | Explained Variance |
| --------- | -----------------: |
| PC1       |             45.60% |
| PC2       |             17.49% |
| **Total** |         **63.08%** |

Therefore, the first two principal components explain approximately:

```
63.08%
```

of the total variance in the customer feature dataset.

This provides a useful two-dimensional representation for visual inspection of the customer clusters.

---

## 14.3 PCA Visualization

The generated PCA figure is:

```
outputs/figures/customer_pca_clusters.png
```

The visualization displays customers according to:

* PC1
* PC2
* K-Means cluster assignment

This provides a visual representation of how the identified customer groups are distributed in the reduced feature space.

---

# 15. Business Segment Naming

Machine learning produces numerical cluster labels such as:

```
Cluster 0
Cluster 1
```

These labels have no inherent business meaning.

Therefore, the cluster profiles are examined and business-friendly names are assigned.

The current mapping is:

| Cluster | Business Segment                        |
| ------: | --------------------------------------- |
|       0 | Occasional / Lower-Engagement Customers |
|       1 | Highly Engaged / High-Value Customers   |

The names are based on the observed behavioral differences in:

* Order volume
* Spending
* Ordering frequency
* Weekend behavior
* Late-night behavior

---

# 16. Final Segmentation Output

The final segmented customer dataset is saved as:

```
data/processed/customer_segments.csv
```

The final dataset contains:

```
5,000 rows
10 columns
```

The columns are:

```
customer_id
total_orders
avg_order_value
total_spending
weekend_orders
late_night_orders
ordering_frequency
avg_rating_given
cluster
segment_name
```

Each customer therefore has:

* Their original behavioral features
* Their assigned K-Means cluster
* Their business-friendly segment name

---

# 17. Final Segment Distribution

The final customer distribution is:

| Segment                                 | Customer Count |
| --------------------------------------- | -------------: |
| Occasional / Lower-Engagement Customers |          2,819 |
| Highly Engaged / High-Value Customers   |          2,181 |

This represents the final customer segmentation used for downstream business analysis.

---

# 18. Generated Outputs

The segmentation module generates the following files.

### Customer Feature Distribution

```
outputs/figures/customer_feature_distributions.png
```

Used to inspect feature distributions.

### Customer Feature Boxplots

```
outputs/figures/customer_feature_boxplots.png
```

Used to inspect potential outliers.

### K-Means Elbow Plot

```
outputs/figures/customer_elbow.png
```

Used to evaluate K-Means inertia across different values of K.

### K-Means Silhouette Plot

```
outputs/figures/customer_silhouette.png
```

Used to compare silhouette scores across different values of K.

### PCA Cluster Visualization

```
outputs/figures/customer_pca_clusters.png
```

Used to visualize the final customer clusters in two dimensions.

### Final Customer Segments

```
data/processed/customer_segments.csv
```

Contains the final customer-level segmentation results.

---

# 19. Segmentation Workflow

The complete segmentation workflow is:

```
customer_features.csv
        │
        ▼
Remove customer_id
        │
        ▼
Customer EDA
        │
        ├── Descriptive Statistics
        ├── Correlation Analysis
        ├── Feature Distributions
        └── Boxplots
        │
        ▼
StandardScaler
        │
        ▼
K-Means Evaluation
        │
        ├── Elbow Method
        └── Silhouette Score
        │
        ▼
Select K = 2
        │
        ▼
Final K-Means
        │
        ▼
Cluster Profiling
        │
        ▼
DBSCAN Evaluation
        │
        ▼
K-Means vs DBSCAN Comparison
        │
        ▼
PCA Visualization
        │
        ▼
Business Segment Naming
        │
        ▼
customer_segments.csv
```

---

# 20. Why K-Means Was Selected

K-Means was selected as the primary segmentation method because it produced the strongest result among the tested approaches.

The main evidence is:

* K-Means silhouette score: **0.2513**
* DBSCAN silhouette score: **0.1293**
* K-Means assigns every customer to a segment.
* DBSCAN classifies **53.86%** of customers as noise under the selected parameters.
* K-Means produces two relatively interpretable business segments.

Therefore, the current project uses **K-Means as the final customer segmentation model**, while DBSCAN is retained as a comparative unsupervised learning technique.

---

# 21. Limitations

The current segmentation has several limitations.

## 21.1 Moderate Silhouette Score

The K-Means silhouette score of:

```
0.2513
```

indicates that the clusters are not extremely well separated.

The segments should therefore be interpreted as **behavioral groupings rather than perfectly distinct customer populations**.

---

## 21.2 Strong Feature Correlations

Some variables are strongly correlated.

For example:

```
total_orders ↔ total_spending = 0.82
```

This means some behavioral dimensions may have overlapping information.

---

## 21.3 Rating Representation

Customers without review activity are represented using:

```
avg_rating_given = 0
```

This means the feature represents both rating behavior and absence of review activity.

---

## 21.4 Synthetic Dataset

The dataset is synthetically generated for educational purposes.

Therefore, the resulting customer segments should not be interpreted as actual Uber Eats customer populations.

---

## 21.5 Only Customer Segmentation

The current implementation focuses on customers.

Restaurant segmentation has not been implemented in this module.

---

# 22. Future Improvements

Future versions could improve customer segmentation by introducing additional behavioral features such as:

* Recency of last order
* Customer lifetime value
* Number of unique restaurants visited
* Cuisine diversity
* Cancellation rate
* Promotion usage
* Average delivery distance
* Customer tenure
* Payment behavior
* Review sentiment

Alternative clustering approaches could also be evaluated, such as:

* Hierarchical clustering
* Gaussian Mixture Models
* Alternative DBSCAN parameters
* HDBSCAN

Feature selection or dimensionality-reduction techniques could also be explored to reduce redundancy among highly correlated variables.

---

# 23. Business Applications

The customer segmentation results can support several business decisions.

## Customer Retention

Highly engaged and high-value customers can be targeted with loyalty and retention strategies.

## Customer Re-Engagement

Lower-engagement customers can receive personalized offers or recommendations designed to increase ordering frequency.

## Marketing Personalization

Different customer segments can receive different campaigns instead of applying the same promotion to the entire customer base.

## Customer Value Analysis

The segmentation can help distinguish customers based on their ordering activity and monetary contribution.

## Behavioral Targeting

Weekend and late-night ordering behavior can help inform time-specific promotions and recommendations.

---

# 24. Final Result

The segmentation module successfully transforms the customer feature dataset into meaningful behavioral customer groups.

The final solution uses:

* **StandardScaler** for feature normalization
* **K-Means** for primary customer segmentation
* **DBSCAN** for comparison and density-based analysis
* **PCA** for dimensionality reduction and visualization
* **Cluster profiling** for behavioral interpretation
* **Business segment naming** for practical use

The final customer segmentation consists of two primary groups:

1. **Occasional / Lower-Engagement Customers**
2. **Highly Engaged / High-Value Customers**

The final output is stored in:

```
data/processed/customer_segments.csv
```

This dataset can now be used by downstream business analysis and machine learning components of the **Uber Eats Operational & Customer Intelligence System**.
