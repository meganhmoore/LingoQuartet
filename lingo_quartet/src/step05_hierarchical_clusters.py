"""
Use BERT embeddings and fasttext embeddings to create hierarchical clusters

Authored by Megan
"""

import pandas as pd
import nltk
from nltk.corpus import stopwords

import numpy as np
from matplotlib import pyplot as plt

from scipy.cluster.hierarchy import ward, dendrogram, fcluster
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score

from sklearn.model_selection import train_test_split
from types import Union


nltk.download("punkt")
nltk.download("stopwords")


def get_bert_embeddings(
    filepath: str = "./data/clean_2020articles_w_title_embeddings.pkl",
):
    """
    Get BERT embeddings that were created in script 03

    Inputs:
        filepath(str): filepath to pickled dataframe with embeddings

    Returns:
        train(pd.DataFrame): the training size subset dataframe
        np_embeddings(np.array): the associated embeddings
    """
    df = pd.read_df(filepath)
    train, test = train_test_split(df, test_size=0.4, random_state=42)
    train["bert_embeddings"] = train["bert_sentence_embedding_title_only"].apply(
        lambda x: x.cpu()
    )
    train["np_embeddings"] = train["bert_embeddings"].apply(lambda x: x.numpy())
    np_embeddings = np.vstack(train["np_embeddings"])
    return train, np_embeddings


def get_fast_text_embeddings(filepath: str):
    """
    Get FastText embeddings similar to above

    Inputs:
        filepath(str): filepath to access fasttext numpy
    """
    # todo need to also load original df that embeddings were created from
    np_embeddings = np.load(filepath)
    return np_embeddings


def ward_hierarchical_clustering(feature_matrix: np.array):
    """
    CORE HIERARCHICAL COMPONENT. Compute the cosine distances so that the ward
    clustering algorithm can iteratively combine observations into the same
    cluster if they are closest together at a given step in the process.

    Inputs:
        feature_matrix(np.array): numpy array of embeddings

    Returns:
        linkage_matrix(np.arrray): n-1 x 4 np matrix where n is the number of
            documents/embeddings. This contains the hierarchical information.
    """
    cosine_distance = 1 - cosine_similarity(feature_matrix)
    linkage_matrix = ward(cosine_distance)
    return linkage_matrix


def plot_hierarchical_clusters(linkage_matrix, data, p=100, figure_size=(8, 12)):
    """
    Plot dendrogram from the linkage matrix to show the relationships between
    articles, this is not super informative given that we are working with so
    many articles, and visuals don't scale very well.

    Inputs:
        linkage_matrix(np.array): result of wards
        data(pd.DataFrame): original data
        p(int): level to show the clusters at
        figure_size(Tuple): for plotting
    """
    # set size
    fig, ax = plt.subplots(figsize=figure_size)
    titles = data["title"].values.tolist()
    # plot dendrogram
    R = dendrogram(
        linkage_matrix, orientation="left", truncate_mode="lastp", p=p, no_plot=True
    )
    temp = {R["leaves"][ii]: titles[ii] for ii in range(len(R["leaves"]))}

    def llf(xx):
        return "{}".format(temp[xx])

    ax = dendrogram(
        linkage_matrix,
        truncate_mode="lastp",
        orientation="left",
        p=p,
        leaf_label_func=llf,
        leaf_font_size=10.0,
    )
    plt.tick_params(axis="x", which="both", bottom="off", top="off", labelbottom="off")
    plt.tight_layout()
    # plt.savefig(f'./data/bert_hierachical_clusters_num_docs_{data.shape[0]}_level_{p}.png', dpi=200)


def get_linkage_matrix(np_embeddings: np.array, data: pd.DataFrame):
    """
    Compute the linkage matrix and plot

    Inputs:
        np_embeddings(np.array): embeddings
        data(pd.DataFrame): original data
    """
    linkage_matrix = ward_hierarchical_clustering(np_embeddings)

    plot_hierarchical_clusters(linkage_matrix, p=100, data=data, figure_size=(18, 30))


def compute_silhouette(
    orig_data: pd.DataFrame, linkage_matrix: np.array, distance: Union[int, float]
):
    """
    Compute the silhouette score to get a sense of how close together clusters
    are internally and how far they are from other clusters (i.e. how
    separable they are).

    Inputs:
        orig_data(pd.DataFrame)
        linkage_matrix(np.array)
        distance(int): distance to cut clusters at

    Returns:
        silhouette_avg(float): average silhouette score for all of the
            clusters when you create the clusters from a cutoff of the given
            distance.
    """
    labels = fcluster(linkage_matrix, distance, criterion="distance")
    silhouette_avg = silhouette_score(orig_data, labels)
    return silhouette_avg


def run_analysis(linkage_matrix: np.array, data: pd.DataFrame, np_embedding: np.array):
    """
    Analyze a range of distance cutoffs on the hierarchy to get a sense of
    which ones may have more separable topics and a reasonable number of
    clusters.

    Inputs:
        linkage_matrix(np.array): hierarchy
        data(pd.DataFrame): original data
        np_embedding(np.array): array of embeddings that created the hierarchy

    Returns:
        train_cluster_assignments(pd.DataFrame): dataframe with all of the
            cluster assignments depending on the distance selected.
        sil_scores(list): silhouette scores for each distance
        num_clusters(list): number of clusters for each distance cutoff
        distance_options(list): each distance cutoff
    """
    train_cluster_assignments = data.copy()
    train_cluster_assignments = train_cluster_assignments.drop(
        ["bert_sentence_embedding_title_only", "bert_embeddings", "np_embeddings"],
        axis=1,
    )
    distance_options = [0.5, 1, 2, 3, 4, 5, 10, 20, 50, 100, 150, 200, 300, 500, 1000]
    num_clusters = []
    for opt in distance_options:
        cluster_assignment = fcluster(linkage_matrix, opt, criterion="distance")
        num_cluster = cluster_assignment.max()
        num_clusters.append(num_cluster)
        print(f"dist: {opt} num_clusters: {num_cluster}")
        train_cluster_assignments[f"cluster_dist_{opt}"] = cluster_assignment

    sil_scores = []
    for opt in distance_options:
        sil_score = compute_silhouette(np_embedding, linkage_matrix, opt)
        sil_scores.append(sil_score)

    for cutoff, score in zip(distance_options, sil_scores):
        print(f"Distance cutoff: {cutoff}, Silhouette Score: {score}")

    return train_cluster_assignments, sil_scores, num_clusters, distance_options


def save_cluster_assignments(
    df: pd.DataFrame,
    filepath: str = "./data/bert_22kdocs_cluster_assignments.csv",
):
    """
    Safe for later analysis
    """
    df.to_csv(filepath)


def plot_sil_vs_clusters(sil_scores: list, num_clusters: list):
    """
    Plot silhouette score against number of clusters

    Inputs:
        sil_scores
        num_clusters
    """
    plt.scatter(sil_scores, num_clusters, marker="o")
    plt.xlabel("Silhouette Score")
    plt.ylabel("Number of Clusters")
    plt.title("Silhouette Score vs. Number of Clusters")
    plt.grid(True)
    plt.show()


def plot_sil_vs_distance(sil_scores: list, distance_options: list):
    """
    Plot silhouette score against distance cutoff

    Inputs:
        sil_scores
        distance_options
    """
    plt.scatter(sil_scores, distance_options, marker="o")
    plt.xlabel("Silhouette Score")
    plt.ylabel("Distance Cutoff")
    plt.title("Silhouette Score vs. Distance Cutoff")
    plt.grid(True)
    plt.show()


def plot_dist_vs_num_clusters(distance_options: list, num_clusters: list):
    """
    Plot distance cutoff against number of clusters

    Inputs:
        distance_options
        num_clusters
    """
    plt.plot(distance_options, num_clusters, marker="o")
    plt.xlabel("Distance Cutoff")
    plt.ylabel("Number of Clusters")
    plt.title("Distance Cutoff vs. Number of Clusters")
    plt.grid(True)
    plt.show()


def cluster_top_words(
    train_cluster_assignments: pd.DataFrame, col: str = "cluster_dist_3"
):
    """
    Use TF-IDF to extract the top words from each cluster in a given cluster
    result (i.e. a chosen distance)

    Inputs:
        train_cluster_assignments(pd.DataFrame)
        col(str): column for cutoff you want to assess the clusters for

    Returns:
        top_words(list): top 5 words for each of the 10 biggest clusters
        count_docs(list): number of documents in each of the 10 biggest
            clusters
    """
    stop_words = set(stopwords.words("english"))

    tfidf_vectorizer = TfidfVectorizer(stop_words=list(stop_words))

    tfidf_scores_by_cluster = {}
    cluster_sizes = {}
    for clust_id in list(train_cluster_assignments.loc[:, col].unique()):
        clust_df = train_cluster_assignments.loc[
            train_cluster_assignments.loc[:, col] == clust_id, :
        ]
        cluster_sizes[clust_id] = clust_df.shape[0]
        clust_text = clust_df.loc[:, "title"].values
        tfidf_matrix = tfidf_vectorizer.fit_transform(clust_text)
        feature_names = tfidf_vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.sum(axis=0)
        tfidf_scores_by_cluster[clust_id] = [
            (feature_names[col], tfidf_scores[0, col])
            for col in tfidf_matrix.nonzero()[1]
        ]

    top_words_by_cluster = {}
    for cluster_id, scores in tfidf_scores_by_cluster.items():
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        unique_sorted_data = []
        seen_tuples = set()
        for key, value in sorted_scores:
            if value not in seen_tuples:
                unique_sorted_data.append((key, value))
                seen_tuples.add(value)
        top_words_by_cluster[cluster_id] = unique_sorted_data[
            :10
        ]  # sorted_scores[:10]  # Select top N words

    sorted_data = sorted(cluster_sizes.items(), key=lambda x: x[1])

    biggest_clusters = sorted_data[-10:]
    top_words = []
    count_docs = []
    for keyval, count in biggest_clusters:
        print(f"top words for cluster {keyval} with {count} articles: \n")
        for val in top_words_by_cluster[keyval][0:5]:
            print(val)
        count_docs.append(count)
        top_words.append(
            "_".join([word[0] for word in top_words_by_cluster[keyval][0:5]])
        )
        print()

    print(top_words)
    return top_words, count_docs


def plot_top_words_top_clusters(top_words: list, count_docs: list):
    """
    Bar plot of top words against count of documents

    Inputs:
        top_words(list)
        count_docs(list)
    """
    plt.barh(top_words, count_docs, color="plum")
    plt.xlabel("Count of Documents")
    plt.ylabel("Topic Words")
    plt.title("Count of Documents per Topic Cluster")
    plt.show()


if __name__ == "__main__":
    data, embeddings = get_bert_embeddings()
    linkage_matrix = get_linkage_matrix(embeddings, data)
    cluster_assignments, sil_scores, num_clusters, distance_options = run_analysis(
        linkage_matrix, data, embeddings
    )
    plot_sil_vs_clusters(sil_scores, num_clusters)
    plot_sil_vs_distance(sil_scores, distance_options)
    plot_dist_vs_num_clusters(distance_options, num_clusters)
    top_words, count_docs = cluster_top_words(cluster_assignments)
    plot_top_words_top_clusters(top_words, count_docs)
