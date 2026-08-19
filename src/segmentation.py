from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SEGMENTS_OUTPUT = PROCESSED_DATA_DIR / "customer_segments.csv"


def prepare_features(feature_file="customer_features.csv"):

    file_path = PROCESSED_DATA_DIR / feature_file

    if not file_path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    if "customer_id" not in df.columns:
        raise ValueError(
            "Expected 'customer_id' column was not found."
        )

    customer_ids = df["customer_id"].copy()

    feature_matrix = df.drop(
        columns=["customer_id"]
    )

    return customer_ids, feature_matrix


def customer_eda(feature_matrix, output_dir):
    """
    Perform lightweight EDA on customer behavioral features.
    """

    print("\n========== CUSTOMER EDA ==========")

    print("\nFeature statistics:")
    print(
        feature_matrix
        .describe()
        .round(2)
        .to_string()
    )

    print("\nCorrelation matrix:")
    print(
        feature_matrix
        .corr()
        .round(2)
        .to_string()
    )

    # Feature distributions
    feature_matrix.hist(
        figsize=(12, 8),
        bins=20
    )

    plt.tight_layout()

    distribution_path = (
        output_dir / "customer_feature_distributions.png"
    )

    plt.savefig(distribution_path)
    plt.close()

    # Boxplots for outlier inspection
    plt.figure(figsize=(12, 6))

    feature_matrix.boxplot()

    plt.xticks(rotation=45)
    plt.tight_layout()

    boxplot_path = (
        output_dir / "customer_feature_boxplots.png"
    )

    plt.savefig(boxplot_path)
    plt.close()

    print("\nEDA figures saved:")
    print(distribution_path)
    print(boxplot_path)


def scale_features(feature_matrix):

    scaler = StandardScaler()

    scaled_array = scaler.fit_transform(feature_matrix)

    scaled_features = pd.DataFrame(
        scaled_array,
        columns=feature_matrix.columns,
        index=feature_matrix.index
    )

    return scaler, scaled_features


def find_optimal_k(
    scaled_features,
    k_range=range(2, 9),
    random_state=42
):
    """
    Evaluate different K values for K-Means clustering.

    Calculates:
        - Inertia for the Elbow Method
        - Silhouette Score for cluster separation

    Also saves:
        - customer_elbow.png
        - customer_silhouette.png

    Returns:
        results: DataFrame containing K, inertia, and silhouette score.
    """

    results = []

    for k in k_range:

        kmeans = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=10
        )

        cluster_labels = kmeans.fit_predict(
            scaled_features
        )

        inertia = kmeans.inertia_

        silhouette = silhouette_score(
            scaled_features,
            cluster_labels
        )

        results.append({
            "k": k,
            "inertia": inertia,
            "silhouette_score": silhouette
        })

    results = pd.DataFrame(results)

    # Create output directory
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    figures_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Elbow Plot
    plt.figure(figsize=(8, 5))

    plt.plot(
        results["k"],
        results["inertia"],
        marker="o"
    )

    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.title("Customer K-Means Elbow Method")
    plt.xticks(list(k_range))
    plt.grid(True)

    plt.tight_layout()

    elbow_path = figures_dir / "customer_elbow.png"

    plt.savefig(
        elbow_path,
        dpi=300
    )

    plt.close()

    # Silhouette Plot
    plt.figure(figsize=(8, 5))

    plt.plot(
        results["k"],
        results["silhouette_score"],
        marker="o"
    )

    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Customer K-Means Silhouette Analysis")
    plt.xticks(list(k_range))
    plt.grid(True)

    plt.tight_layout()

    silhouette_path = (
        figures_dir / "customer_silhouette.png"
    )

    plt.savefig(
        silhouette_path,
        dpi=300
    )

    plt.close()

    print("\n========== K-MEANS K SELECTION ==========")

    print(results.round(4).to_string(index=False))

    print(f"\nElbow plot saved to: {elbow_path}")
    print(
        f"Silhouette plot saved to: {silhouette_path}"
    )

    return results


def run_kmeans(
    feature_matrix,
    scaled_features,
    n_clusters,
    random_state=42
):
    """
    Train the final K-Means model and assign cluster labels.

    Parameters:
        feature_matrix: Original unscaled behavioral features.
        scaled_features: Standardized features used by K-Means.
        n_clusters: Selected number of clusters.
        random_state: Seed for reproducibility.

    Returns:
        model: Fitted KMeans model.
        clustered_data: Original features with cluster labels.
    """

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )

    cluster_labels = model.fit_predict(
        scaled_features
    )

    clustered_data = feature_matrix.copy()

    clustered_data["cluster"] = cluster_labels

    return model, clustered_data


def profile_clusters(clustered_data):
    """
    Create a behavioral profile for each customer cluster.

    Returns:
        cluster_profile: DataFrame containing average behavioral
        characteristics for each cluster.
    """

    profile_columns = [
        "total_orders",
        "avg_order_value",
        "total_spending",
        "ordering_frequency",
        "avg_rating_given",
        "weekend_orders",
        "late_night_orders",
    ]

    cluster_profile = (
        clustered_data
        .groupby("cluster")[profile_columns]
        .mean()
        .round(2)
    )

    customer_counts = (
        clustered_data["cluster"]
        .value_counts()
        .sort_index()
        .rename("customer_count")
    )

    cluster_profile = (
        customer_counts.to_frame()
        .join(cluster_profile)
    )

    return cluster_profile


def assign_segment_names(clustered_data):
    """
    Assign business-friendly names to customer clusters.

    The mapping is based on the observed cluster profile:
        Cluster 0 -> Lower engagement and lower spending
        Cluster 1 -> Higher engagement and higher spending

    Returns:
        DataFrame with a segment_name column.
    """

    segment_mapping = {
        0: "Occasional / Lower-Engagement Customers",
        1: "Highly Engaged / High-Value Customers",
    }

    clustered_data = clustered_data.copy()

    clustered_data["segment_name"] = (
        clustered_data["cluster"]
        .map(segment_mapping)
    )

    return clustered_data


def run_dbscan(
    scaled_features,
    eps=0.8,
    min_samples=10
):
    """
    Run DBSCAN clustering on scaled customer features.
    """

    model = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )

    cluster_labels = model.fit_predict(
        scaled_features
    )

    return model, cluster_labels


def evaluate_dbscan(
    scaled_features,
    eps_values=(0.5, 0.7, 0.8, 0.9, 1.1, 1.3),
    min_samples=10
):

    results = []

    for eps in eps_values:

        model = DBSCAN(
            eps=eps,
            min_samples=min_samples
        )

        labels = model.fit_predict(
            scaled_features
        )

        n_clusters = len(
            set(labels) - {-1}
        )

        noise_points = (
            labels == -1
        ).sum()

        results.append({
            "eps": eps,
            "clusters": n_clusters,
            "noise_points": noise_points
        })

    results = pd.DataFrame(results)

    print("\n========== DBSCAN PARAMETER EVALUATION ==========")
    print(results.to_string(index=False))

    return results


def compare_clustering_methods(
    scaled_features,
    kmeans_labels,
    dbscan_labels
):
    """
    Compare K-Means and DBSCAN clustering results.
    """

    comparison = []

    # K-Means
    kmeans_clusters = len(
        set(kmeans_labels)
    )

    kmeans_silhouette = silhouette_score(
        scaled_features,
        kmeans_labels
    )

    comparison.append({
        "method": "K-Means",
        "clusters": kmeans_clusters,
        "silhouette_score": round(
            kmeans_silhouette, 4
        ),
        "noise_points": 0,
        "noise_percentage": 0.0
    })

    # DBSCAN
    dbscan_cluster_labels = (
        set(dbscan_labels) - {-1}
    )

    dbscan_clusters = len(
        dbscan_cluster_labels
    )

    noise_points = (
        dbscan_labels == -1
    ).sum()

    non_noise_mask = (
        dbscan_labels != -1
    )

    if dbscan_clusters > 1:
        dbscan_silhouette = silhouette_score(
            scaled_features[non_noise_mask],
            dbscan_labels[non_noise_mask]
        )
    else:
        dbscan_silhouette = None

    comparison.append({
        "method": "DBSCAN",
        "clusters": dbscan_clusters,
        "silhouette_score": (
            round(dbscan_silhouette, 4)
            if dbscan_silhouette is not None
            else None
        ),
        "noise_points": noise_points,
        "noise_percentage": round(
            noise_points / len(dbscan_labels) * 100,
            2
        )
    })

    comparison = pd.DataFrame(
        comparison
    )

    print("\n========== CLUSTERING COMPARISON ==========")
    print(
        comparison.to_string(index=False)
    )

    return comparison


def apply_pca(
    scaled_features,
    cluster_labels,
    output_dir
):
    """
    Reduce customer features to two principal components
    for cluster visualization.
    """

    pca = PCA(
        n_components=2,
        random_state=42
    )

    pca_features = pca.fit_transform(
        scaled_features
    )

    pca_data = pd.DataFrame(
        pca_features,
        columns=["PC1", "PC2"]
    )

    pca_data["cluster"] = cluster_labels

    explained_variance = (
        pca.explained_variance_ratio_ * 100
    )

    print("\n========== PCA ==========")

    print(
        f"PC1 explained variance: "
        f"{explained_variance[0]:.2f}%"
    )

    print(
        f"PC2 explained variance: "
        f"{explained_variance[1]:.2f}%"
    )

    print(
        f"Total explained variance: "
        f"{explained_variance.sum():.2f}%"
    )

    plt.figure(figsize=(8, 6))

    for cluster in sorted(
        set(cluster_labels)
    ):
        cluster_data = pca_data[
            pca_data["cluster"] == cluster
        ]

        plt.scatter(
            cluster_data["PC1"],
            cluster_data["PC2"],
            label=f"Cluster {cluster}",
            alpha=0.6
        )

    plt.xlabel(
        f"PC1 ({explained_variance[0]:.1f}%)"
    )

    plt.ylabel(
        f"PC2 ({explained_variance[1]:.1f}%)"
    )

    plt.title(
        "Customer Segments - PCA Visualization"
    )

    plt.legend()
    plt.tight_layout()

    pca_path = (
        output_dir /
        "customer_pca_clusters.png"
    )

    plt.savefig(
        pca_path,
        dpi=300
    )

    plt.close()

    print(
        f"PCA plot saved to: {pca_path}"
    )

    return pca, pca_data


if __name__ == "__main__":

    # --------------------------------------------------
    # 1. Prepare customer features
    # --------------------------------------------------

    customer_ids, feature_matrix = prepare_features()


    # --------------------------------------------------
    # 2. Customer EDA
    # --------------------------------------------------

    customer_eda(
        feature_matrix,
        FIGURES_DIR
    )


    # --------------------------------------------------
    # 3. Scale features
    # --------------------------------------------------

    scaler, scaled_features = scale_features(
        feature_matrix
    )


    # --------------------------------------------------
    # 4. Select K for K-Means
    # --------------------------------------------------

    results = find_optimal_k(
        scaled_features
    )

    selected_k = 2


    # --------------------------------------------------
    # 5. Final K-Means
    # --------------------------------------------------

    model, clustered_data = run_kmeans(
        feature_matrix,
        scaled_features,
        selected_k
    )

    clustered_data.insert(
        0,
        "customer_id",
        customer_ids
    )

    print("\n========== FINAL K-MEANS ==========")

    print(f"Selected K: {selected_k}")

    print("\nCluster counts:")

    print(
        clustered_data["cluster"]
        .value_counts()
        .sort_index()
    )


    # --------------------------------------------------
    # 6. Evaluate DBSCAN parameters
    # --------------------------------------------------

    dbscan_results = evaluate_dbscan(
        scaled_features
    )


    # --------------------------------------------------
    # 7. Run DBSCAN
    # --------------------------------------------------

    dbscan_model, dbscan_labels = run_dbscan(
        scaled_features,
        eps=0.8,
        min_samples=10
    )

    print("\n========== DBSCAN ==========")

    print(
        "Cluster labels:",
        sorted(set(dbscan_labels))
    )

    print(
        "Number of clusters:",
        len(set(dbscan_labels)) - (
            1 if -1 in dbscan_labels else 0
        )
    )

    print(
        "Noise points:",
        (dbscan_labels == -1).sum()
    )


    # --------------------------------------------------
    # 8. Compare K-Means and DBSCAN
    # --------------------------------------------------

    comparison = compare_clustering_methods(
        scaled_features,
        model.labels_,
        dbscan_labels
    )


    # --------------------------------------------------
    # 9. PCA visualization
    # --------------------------------------------------

    pca_model, pca_data = apply_pca(
        scaled_features,
        model.labels_,
        FIGURES_DIR
    )


    # --------------------------------------------------
    # 10. Profile K-Means clusters
    # --------------------------------------------------

    cluster_profile = profile_clusters(
        clustered_data
    )

    print("\n========== CLUSTER PROFILE ==========")

    print(
        cluster_profile.to_string()
    )


    # --------------------------------------------------
    # 11. Assign business segment names
    # --------------------------------------------------

    clustered_data = assign_segment_names(
        clustered_data
    )

    print("\n========== BUSINESS SEGMENTS ==========")

    print(
        clustered_data[
            [
                "customer_id",
                "cluster",
                "segment_name"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


    # --------------------------------------------------
    # 12. Save final customer segments
    # --------------------------------------------------

    clustered_data.to_csv(
        SEGMENTS_OUTPUT,
        index=False
    )

    print(
        f"\nSegmented customer data saved to: "
        f"{SEGMENTS_OUTPUT}"
    )