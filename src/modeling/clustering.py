import os
import pandas as pd

from torch.utils.data import DataLoader

from collections import Counter
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import umap


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


from ..data import SheepDataset, get_valid_transforms, load_pseudo_labels
from .feature_extraction import extract_features
from .helpers import load_models

from src import CONFIG, Logger

logger = Logger()


class KMeansClustering:
    def __init__(self, pseudo_df, train_df, output_dir, purity_threshold=0.9):
        self.config = CONFIG
        self.train_df = train_df
        self.output_dir = output_dir
        self.pseudo_df = pseudo_df
        self.purity_threshold = purity_threshold

        self.models = self._load_models()
        self.test_loader = self._load_test_loader()
        self.filenames = self.test_loader.dataset.img_files

    def _load_models(self):
        model_files = sorted(
            [f for f in os.listdir(self.config.models_dir) if f.endswith(".pth")]
        )
        return load_models(model_files)

    def _load_test_loader(self):
        files = sorted(
            [f for f in os.listdir(self.config.test_dir) if f.lower().endswith(".jpg")]
        )
        dataset = SheepDataset(
            image_dir=self.config.test_dir,
            transform=get_valid_transforms(),
            is_test=True,
        )
        dataset.img_files = files
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    def run(self):
        logger.info(f"{len(self.filenames)} test images loaded.")
        logger.info(f"Using {len(self.models)} ensemble models.")

        features, filenames = extract_features(self.models, self.test_loader)

        # Run clustering
        embedding, cluster_labels = run_clustering(features, k=self.config.num_classes)

        # Load pseudo-labels
        pseudo_label_map, pseudo_conf_map = load_pseudo_labels(self.pseudo_df)

        predicted_label_names = get_cluster_labels_from_pseudo(
            cluster_labels, filenames, pseudo_label_map
        )

        df_clusters = pd.DataFrame(
            {
                "filename": filenames,
                "cluster": cluster_labels,
                "u1": embedding[:, 0],
                "u2": embedding[:, 1],
                "pred_label": predicted_label_names,
                "pconf": [pseudo_conf_map.get(f, np.nan) for f in filenames],
            }
        )
        # Visualize the clusters
        show_clusters(df_clusters, output_dir=self.output_dir)

        df_clusters.to_csv(
            os.path.join(self.output_dir, "clustered_test_results.csv"), index=False
        )
        logger.info(
            f"Clustered CSV saved to {os.path.join(self.output_dir, 'clustered_test_results.csv')}"
        )

        logger.info("Calculating cluster purity...")
        purity_map, cluster_label_map = calc_cluster_purity(df_clusters)

        logger.info("Building merged CSV...")
        merged_df = build_csv(
            train_df=self.train_df,
            cluster_df=df_clusters,
            purity_map=purity_map,
            label_map=cluster_label_map,
            feats=features,
            output_dir=self.output_dir,
            purity_threshold=self.purity_threshold,
            return_df=True,
        )
        logger.info(
            f"Merged CSV saved to {os.path.join(self.output_dir, 'pseudo_clustered_merged.csv')}"
        )

        return df_clusters, merged_df


def run_clustering(features, k=7):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    reducer = umap.UMAP(
        n_components=2, random_state=CONFIG.seed, n_neighbors=15, min_dist=0.1
    )
    embedding = reducer.fit_transform(scaled)
    clusterer = KMeans(n_clusters=k, random_state=CONFIG.seed)
    cluster_labels = clusterer.fit_predict(scaled)
    return embedding, cluster_labels


def get_cluster_labels_from_pseudo(cluster_labels, filenames, pseudo_map):
    """
    Generally the cluster labels are organized as `cluster_1`, `cluster_2`, etc.
    This function will map the cluster labels to the pseudo labels, and return the label names.
    """
    cluster_map = {}
    unique_clusters = np.unique(cluster_labels)
    logger.info(
        f"Number of clusters found: {len(unique_clusters[unique_clusters >= 0])}"
    )

    pseudo_labels = [pseudo_map.get(f, None) for f in filenames]

    for c in unique_clusters:
        if c == -1:
            cluster_map[c] = "noise"
            continue
        # Get pseudo labels of samples in this cluster
        plabels = [
            pseudo_labels[i]
            for i in range(len(filenames))
            if cluster_labels[i] == c and pseudo_labels[i] is not None
        ]
        cluster_map[c] = Counter(plabels).most_common(1)[0][0] if plabels else "unknown"

    label_names = [cluster_map.get(c, "unknown") for c in cluster_labels]
    return label_names


def show_clusters(df_clusters, output_dir=None):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df_clusters,
        x="u1",
        y="u2",
        hue="pred_label",
        palette="tab10",
        s=20,
        alpha=0.8,
    )
    plt.title("UMAP projection by pseudo-label")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, "cluster_umap.png"))
    plt.show()


def calc_cluster_purity(df):
    purity, label_map = {}, {}
    for c in df["cluster"].unique():
        if c == -1:
            continue
        group = df[df["cluster"] == c]["pred_label"]
        freq = group.value_counts(normalize=True)
        purity[c] = freq.max()
        label_map[c] = freq.idxmax()
    return purity, label_map


def filter_core_samples(df, features, pct=0.5):
    df = df[df["cluster"] != -1].copy()
    selected = []
    for c in df["cluster"].unique():
        idxs = df[df["cluster"] == c].index
        feats = features[idxs]
        dists = cdist(feats, feats.mean(axis=0, keepdims=True)).flatten()
        cutoff = np.quantile(dists, pct)
        selected.extend(i for i, d in zip(idxs, dists) if d <= cutoff)
    return df.loc[selected].copy()


def build_csv(
    train_df,
    cluster_df,
    purity_map,
    label_map,
    feats,
    output_dir,
    purity_threshold=0.9,
    return_df=False,
):
    logger.info(f"Selected purity threshold: {purity_threshold}")

    # Original train data
    train = train_df[["filename", "label"]].copy()
    train["conf"] = 1.0
    train["src"] = "train"
    logger.info(f"Original train samples: {len(train)}")

    # Pseudo labeled data
    pseudo = cluster_df[~cluster_df["pconf"].isna()].copy()
    pseudo = pseudo[["filename", "pred_label", "pconf"]].rename(
        columns={"pred_label": "label", "pconf": "conf"}
    )
    pseudo["src"] = "pseudo"
    logger.info(f"Pseudo labeled samples: {len(pseudo)}")

    # Cluster data filtered by purity threshold and core samples
    cluster = cluster_df[cluster_df["pconf"].isna()].copy()
    cluster = cluster[cluster["cluster"].map(purity_map) >= purity_threshold].copy()
    cluster = cluster[cluster["cluster"].map(label_map) != "unknown"]
    cluster = cluster[cluster["cluster"].map(label_map) != "noise"]
    cluster = filter_core_samples(cluster, feats)
    cluster["label"] = cluster["cluster"].map(label_map)
    cluster["conf"] = purity_threshold
    cluster["src"] = "cluster"
    cluster = cluster[["filename", "label", "conf", "src"]]
    logger.info(
        f"Cluster samples after purity filtering and core sample filtering: {len(cluster)}"
    )

    # Merge all
    merged_df = pd.concat([train, pseudo, cluster], ignore_index=True)
    merged_df.to_csv(
        os.path.join(output_dir, "pseudo_clustered_merged.csv"), index=False
    )
    logger.info(
        f"Merged CSV saved to {os.path.join(output_dir, 'pseudo_clustered_merged.csv')}"
    )
    logger.info(f"Total samples after merging: {len(merged_df)}")

    if return_df:
        return merged_df
