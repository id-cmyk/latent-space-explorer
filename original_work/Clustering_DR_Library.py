# Library
import os
import time

# Visualization
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap

# Clustering
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

# Dimensionality Reduction
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


# Get feature Data
def get_filepaths(directory):
    file_data = []
    for root, directories, files in os.walk(directory):  # walk through the directory to find the data files
        for filename in files:
            if filename.endswith(".pt"):  # Search for .pt tensor files in the file
                filepath = os.path.join(root, filename)
                features = torch.load(filepath, map_location=torch.device('cpu'))
                label = os.path.basename(root)
                file_data.append((label, features))  # Get a tuple containing feature data and class names
    return file_data


feature_data = get_filepaths("E:\pythonProject\Features")

# Get data and labels separately and convert them to an array to see the shape of data
lbl = []
for i in feature_data:
    lbl.append(i[0])
classes = np.array(lbl)  # Create an array of class names.

data = []
for i in feature_data:
    data.append(i[1])
feature_array = np.array(data)  # Create an array of feature data


## CLUSTERING
# KMeans Clustering
def KMEANS_CL(k_clusters, k_cluster_input):
    kmeans = KMeans(init="k-means++", n_clusters=k_clusters, n_init=10, random_state=42)

    clusters = kmeans.fit(k_cluster_input)
    cluster_labels = clusters.labels_
    centroids = clusters.cluster_centers_
    silhouette_avg = silhouette_score(feature_array, cluster_labels)
    print("The average silhouette_score is :", silhouette_avg )

    return [cluster_labels, centroids]

#Spectral Clustering
def SPECTRAL_CL(spectral_n_clusters, spectral_cluster_input):
    spectral = SpectralClustering(n_clusters=spectral_n_clusters, affinity='rbf', assign_labels='cluster_qr',
                                  n_components=None, n_init=10, n_jobs=None,
                                  n_neighbors=10, random_state=0)
    sc = spectral.fit(spectral_cluster_input)
    labels = sc.labels_
    silhouette_avg = silhouette_score(feature_array, labels)
    print("The average silhouette_score is :", silhouette_avg )

    return labels



## DIMENSIONALITY REDUCTION

# 1.  Dimensionality Reduction
def PCA_DR(n_pca, pca_input):
    start_time = time.time()
    pca = PCA(n_pca, random_state=0)  # Selecting how many axis or Principal components I need.
    PCA_components = pca.fit_transform(pca_input)
    variance_ratio = pca.explained_variance_ratio_
    elapsed_time = (time.time() - start_time)

    return variance_ratio, PCA_components, elapsed_time

#Plot the graph for PCA
def plot_PCA(components, variance_ratio, used_time):
    plt.scatter(components[:, 0], components[:, 1], c='red')
    plt.title(f"PCA Dimension Reduction --- used_time :{used_time:.2f}s")
    percentage_variance = np.round(variance_ratio * 100, decimals=1)
    plt.xlabel(f"PC1 with {percentage_variance[0]} % variance")
    plt.ylabel(f"PC1 with {percentage_variance[1]} % variance")
    plt.show()

#2. LDA Dimensionality Reduction
def LDA_DR(n_lda, lda_data, lda_classes):
    start_time = time.time()
    lda = LinearDiscriminantAnalysis(solver='svd', n_components=n_lda)  # Use 2 components for visualization
    lda_components = lda.fit_transform(lda_data, lda_classes)
    elapsed_time = (time.time() - start_time)
    return lda_components, elapsed_time

#Plot the graph for the LDA
def plot_LDA(x1, x2, used_time):
    plt.scatter(x1, x2, c='red')
    plt.title(f"LDA Dimension Reduction --- used_time :{used_time:.2f}s")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()

#3. t-SNE Dimensionality Reduction
def TSNE_DR(n_tsne, tsne_input):
    start_time = time.time()
    X_embedded = TSNE(n_tsne, learning_rate='auto', init='random', perplexity=8)
    embeddings = X_embedded.fit_transform(tsne_input)
    elapsed_time = (time.time() - start_time)
    return embeddings, elapsed_time

#Plot the graph for the t-SNE
def plot_TSNE(x1, x2, used_time):
    plt.scatter(x1, x2, c='red')
    plt.title(f"t-SNE Dimension Reduction --- used_time :{used_time:.2f}s")
    plt.xlabel("Embedding 1")
    plt.ylabel("Embedding 2")
    plt.show()

#4. UMAP Dimensionality Reduction
def UMAP_DR(n_umap, umap_input):
    start_time = time.time()
    reducer = umap.UMAP(n_neighbors=15, n_components=n_umap,learning_rate=1.0, init='spectral')
    embedding = reducer.fit_transform(umap_input)
    elapsed_time = (time.time() - start_time)
    return embedding, elapsed_time

#Plot the graph for the UMAP
def plot_UMAP(x1, x2, used_time):
    plt.scatter(x1, x2, c='blue')
    plt.title(f"UMAP Dimension Reduction --- used_time :{used_time:.2f}s")
    plt.xlabel("Embedding 1")
    plt.ylabel("Embedding 2")
    plt.show()

##DIMENSIONALITY REDUCTION BEFORE CLUSTERING

#1. PCA Dimensionality Reduction with KMeans Clustering
def PCA_kmeans(n_pca, k_clusters, pca_input):
    start_time = time.time()
    pca_for_kmeans = PCA_DR(n_pca, pca_input)
    kmeans_for_pca = KMEANS_CL(k_clusters, pca_for_kmeans[1])
    elapsed_time = (time.time() - start_time)
    return kmeans_for_pca, pca_for_kmeans[1], elapsed_time

#Plot the graph for PCA Dimensionality Reduction with KMeans Clustering
# plot the clusters
def plot_PCA_kmeans(pca_components_kmeans, kmeans_cluster_labels_after_PCA, kmeans_centroids_after_PCA, used_time):
    plt.scatter(pca_components_kmeans[:, 0], pca_components_kmeans[:, 1], c=kmeans_cluster_labels_after_PCA, s=60)
    plt.scatter(kmeans_centroids_after_PCA[:, 0], kmeans_centroids_after_PCA[:, 1], c='black', marker='+', s=20)
    plt.title(f"PCA Dimension Reduction and k means clustering --- used_time :{used_time:.2f}s")
    plt.xlabel("PC1 ")
    plt.ylabel("PC2")
    plt.show()


#2. LDA Dimensionality Reduction with KMeans Clustering

def LDA_kmeans(n_lda, lda_data, lda_classes,k_clusters):  # (LDA components, no.of clusters, lda input features, lda classes, kmeans input data)
    start_time = time.time()
    lda_out_for_kmeans = LDA_DR(n_lda, lda_data, lda_classes)
    kmeans_for_lda = KMEANS_CL(k_clusters, lda_out_for_kmeans[0])  # cluster input = DR output
    elapsed_time = (time.time() - start_time)
    return kmeans_for_lda, lda_out_for_kmeans[0], elapsed_time

#Plot the graph for LDA Dimensionality Reduction with KMeans Clustering
def plot_LDA_kmeans(x1_lda_kmeans, x2_lda_kmeans, kmeans_cluster_labels_after_LDA, kmeans_centroids_after_LDA,
                    used_time):
    plt.scatter(x1_lda_kmeans, x2_lda_kmeans, c=kmeans_cluster_labels_after_LDA, s=60)
    plt.scatter(kmeans_centroids_after_LDA[:, 0], kmeans_centroids_after_LDA[:, 1], c='black', marker='+', s=20)
    plt.title(f"LDA Dimension Reduction and k means clustering --- used_time :{used_time:.2f}s")
    plt.xlabel("LDA1 ")
    plt.ylabel("LDA2")
    plt.show()

#3. t-SNE Dimensionality Reduction with KMeans Clustering
def tSNE_kmeans(n_tsne, tsne_input, k_clusters):
    start_time = time.time()  # (TSNE embeddings, no.of clusters, TSNE input data, kmeans input data)
    tSNE_for_kmeans = TSNE_DR(n_tsne, tsne_input)
    kmeans_for_tSNE = KMEANS_CL(k_clusters, tSNE_for_kmeans[0])
    elapsed_time = (time.time() - start_time)

    return kmeans_for_tSNE, tSNE_for_kmeans[0], elapsed_time

#plot the graph for t-SNE
# Dimensionality Reduction with KMeans Clustering
# plot the clusters
def plot_tSNE_kmeans(x1_tsne_kmeans, x2_tsne_kmeans, kmeans_cluster_labels_after_tSNE, kmeans_centroids_after_tSNE,
                     used_time):
    plt.scatter(x1_tsne_kmeans, x2_tsne_kmeans, c=kmeans_cluster_labels_after_tSNE, s=20)
    plt.scatter(kmeans_centroids_after_tSNE[:, 0], kmeans_centroids_after_tSNE[:, 1], c='red', marker='X', s=60)
    plt.title(f"tSNE Dimension Reduction and k means clustering --- used_time :{used_time:.2f}s")
    plt.xlabel("tSNE1 ")
    plt.ylabel("tSNE2")
    plt.show()


#4. UMAP Dimensionality Reduction with KMeans Clustering
def UMAP_kmeans(n_umap, umap_input,
                k_clusters):  # (no of UMAP embeddings, no.of clusters, TSNE input data, kmeans input data)
    start_time = time.time()
    UMAP_for_kmeans = UMAP_DR(n_umap, umap_input)
    kmeans_for_UMAP = KMEANS_CL(k_clusters, UMAP_for_kmeans[0])
    elapsed_time = (time.time() - start_time)
    return kmeans_for_UMAP, UMAP_for_kmeans[0], elapsed_time

#Plot the graph for the UMAP Dimensionality Reduction with KMeans Clustering
def plot_UMAP_kmeans(x1_umap_kmeans, x2_umap_kmeans, kmeans_cluster_labels_after_umap, kmeans_centroids_after_umap,
                     used_time):
    plt.scatter(x1_umap_kmeans, x2_umap_kmeans, c=kmeans_cluster_labels_after_umap, s=20)
    plt.scatter(kmeans_centroids_after_umap[:, 0], kmeans_centroids_after_umap[:, 1], c='red', marker='X', s=60)
    plt.title(f"UMAP Dimension Reduction and k means clustering --- used_time :{used_time:.2f}s")
    plt.xlabel("UMAP1 ")
    plt.ylabel("UMAP2")
    plt.show()

#5. PCA Dimensionality Reduction with Spectral Clustering

def PCA_Spectral(n_pca, spectral_n_clusters,pca_input):  # (PCA components, no.of clusters, PCA input features, kmeans input data)
    start_time = time.time()
    pca_for_spectral = PCA_DR(n_pca, pca_input)
    spectral_for_pca = SPECTRAL_CL(spectral_n_clusters, pca_for_spectral[1])
    elapsed_time = (time.time() - start_time)
    return spectral_for_pca, pca_for_spectral[1], elapsed_time

#plot the graph for  PCA Dimensionality Reduction with Spectral Clustering
# plot the clusters
def plot_PCA_Spectral(pca_components_spectral, spectral_cluster_labels_after_PCA, used_time):
    plt.scatter(pca_components_spectral[:, 0], pca_components_spectral[:, 1], c=spectral_cluster_labels_after_PCA, s=60)
    plt.title(f"PCA Dimension Reduction and Spectral clustering --- used_time :{used_time:.2f}s")

    plt.xlabel("PC1 ")
    plt.ylabel("PC2")
    plt.show()

# 6. LDA with Spectral Clustering
def LDA_Spectral(n_lda, lda_data, lda_classes,spectral_n_clusters):  # (LDA components, no.of clusters, lda input features, lda classes, kmeans input data)
    start_time = time.time()
    lda_out_for_spectral = LDA_DR(n_lda, lda_data, lda_classes)
    spectral_for_lda = SPECTRAL_CL(spectral_n_clusters, lda_out_for_spectral[0])  # cluster input = DR output
    elapsed_time = (time.time() - start_time)
    return spectral_for_lda, lda_out_for_spectral[0], elapsed_time


#plot the graph for LDA Dimensionality Reduction with Spectral Clustering
def plot_LDA_Spectral(x1_lda_spectral, x2_lda_spectral, spectral_cluster_labels_after_LDA, used_time):
    plt.scatter(x1_lda_spectral, x2_lda_spectral, c=spectral_cluster_labels_after_LDA, s=60)
    plt.title(f"LDA Dimension Reduction and Spectral clustering --- used_time :{used_time:.2f}s")
    plt.xlabel("LDA1 ")
    plt.ylabel("LDA2")
    plt.show()


#7. Spectral with TSNE
def tSNE_Spectral(n_tsne, tsne_input,
                  spectral_n_clusters):  # (TSNE embeddings, no.of clusters, TSNE input data, kmeans input data)
    start_time = time.time()
    tSNE_for_spectral = TSNE_DR(n_tsne, tsne_input)
    spectral_for_tSNE = SPECTRAL_CL(spectral_n_clusters, tSNE_for_spectral[0])
    elapsed_time = (time.time() - start_time)
    return spectral_for_tSNE, tSNE_for_spectral[0], elapsed_time


# plot the clusters
def plot_tSNE_Spectral(x1_tsne_spectral, x2_tsne_spectral, spectral_cluster_labels_after_tSNE, used_time):
    plt.scatter(x1_tsne_spectral, x2_tsne_spectral, c=spectral_cluster_labels_after_tSNE, s=20)
    plt.title(f"tSNE Dimension Reduction and Spectral clustering--- used_time :{used_time:.2f}s")
    plt.xlabel("tSNE1 ")
    plt.ylabel("tSNE2")
    plt.show()


#8. Spectral with UMAP
def UMAP_Spectral(n_umap, umap_input,
                  spectral_n_clusters):  # (no of UMAP embeddings, no.of clusters, TSNE input data, kmeans input data)
    start_time = time.time()
    UMAP_for_spectral = UMAP_DR(n_umap, umap_input)
    spectral_for_UMAP = SPECTRAL_CL(spectral_n_clusters, UMAP_for_spectral[0])
    elapsed_time = (time.time() - start_time)
    return spectral_for_UMAP, UMAP_for_spectral[0], elapsed_time


# plot the clusters
def plot_UMAP_Spectral(x1_umap_spectral, x2_umap_spectral, spectral_cluster_labels_after_umap, used_time):
    plt.scatter(x1_umap_spectral, x2_umap_spectral, c=spectral_cluster_labels_after_umap, s=20)
    plt.title(f"UMAP Dimension Reduction and Spectral clustering--- used_time :{used_time:.2f}s")
    plt.xlabel("UMAP1 ")
    plt.ylabel("UMAP2")
    plt.show()


## DIMENSIONALITY REDUCTION AFTER CLUSTERING

# 1. KMeans Clustering before PCA
def kmeans_PCA(n_pca, k_clusters, k_cluster_input):
    start_time = time.time()
    kmeans_for_pca = KMEANS_CL(k_clusters,k_cluster_input)
    pca_for_kmeans = PCA_DR(n_pca,k_cluster_input)
    elapsed_time = (time.time() - start_time)
    kmeans_labels = kmeans_for_pca[0]
    return pca_for_kmeans[1], kmeans_labels,elapsed_time

def plot_kmeans_PCA(PCA_Components,K_Labels,used_time):
    plt.scatter(PCA_Components[:, 0], PCA_Components[:, 1], c =K_Labels, s=60)
    plt.title(f" k means clustering and PCA Dimension Reduction--- used_time :{used_time:.2f}s")
    plt.xlabel("PC1 ")
    plt.ylabel("PC2")
    plt.show()

#2. KMeans Clustering before PCA

def kmeans_LDA(n_lda, k_cluster_input, lda_classes,k_clusters):  # (LDA components, no.of clusters, lda input features, lda classes, kmeans input data)
    start_time = time.time()
    kmeans_for_lda = KMEANS_CL(k_clusters,k_cluster_input )
    lda_for_kmeans = LDA_DR(n_lda, k_cluster_input, lda_classes)
    kMeans_labels = kmeans_for_lda[0]
    elapsed_time = (time.time() - start_time)
    return lda_for_kmeans[0],kMeans_labels, elapsed_time

def plot_kmeans_LDA(x1_lda_kmeans, x2_lda_kmeans, K_Labels,used_time):
    plt.scatter(x1_lda_kmeans, x2_lda_kmeans, c=K_Labels, s=60)
    plt.title(f"k means clustering and LDA Dimension Reduction --- used_time :{used_time:.2f}s")
    plt.xlabel("LDA1 ")
    plt.ylabel("LDA2")
    plt.show()

#3. KMeans Clustering before t-SNE
def kmeans_tSNE(n_tsne,k_clusters,k_cluster_input):
    start_time = time.time()  # (TSNE embeddings, no.of clusters, TSNE input data, kmeans input data)
    kmeans_for_tSNE = KMEANS_CL(k_clusters, k_cluster_input)
    tSNE_for_kmeans = TSNE_DR(n_tsne, k_cluster_input)
    kmeans_labels = kmeans_for_tSNE[0]
    elapsed_time = (time.time() - start_time)

    return tSNE_for_kmeans[0],kmeans_labels, elapsed_time



#plot the graph for t-SNE
def plot_kmeans_tSNE(x1_tsne_kmeans, x2_tsne_kmeans,K_Labels,used_time):
    plt.scatter(x1_tsne_kmeans, x2_tsne_kmeans, c=K_Labels, s=20)
    plt.title(f"tSNE Dimension Reduction and k means clustering --- used_time :{used_time:.2f}s")
    plt.xlabel("tSNE1 ")
    plt.ylabel("tSNE2")
    plt.show()


#4. KMeans Clustering before UMAP
def kmeans_UMAP(n_umap,k_clusters,k_cluster_input):  # (no of UMAP embeddings, no.of clusters, TSNE input data, kmeans input data)
    start_time = time.time()
    kmeans_for_UMAP = KMEANS_CL(k_clusters, k_cluster_input)
    UMAP_for_kmeans = UMAP_DR(n_umap, k_cluster_input)
    kmeans_labels = kmeans_for_UMAP[0]
    elapsed_time = (time.time() - start_time)
    return UMAP_for_kmeans[0],kmeans_labels, elapsed_time

 # Plot the graph for the UMAP Dimensionality Reduction with KMeans Clustering
def plot_kmeans_UMAP(x1_umap_kmeans, x2_umap_kmeans, K_Labels,used_time):
    plt.scatter(x1_umap_kmeans, x2_umap_kmeans, c=K_Labels, s=20)
    plt.title(f"UMAP Dimension Reduction after k means clustering --- used_time :{used_time:.2f}s")
    plt.xlabel("UMAP1 ")
    plt.ylabel("UMAP2")
    plt.show()


#5. Spectral Clustering before PCA
def Spectral_PCA(n_pca, spectral_n_clusters,spectral_cluster_input):  # (PCA components, no.of clusters, PCA input features, kmeans input data)
    start_time = time.time()
    spectral_labels = SPECTRAL_CL(spectral_n_clusters, spectral_cluster_input)
    pca_for_spectral = PCA_DR(n_pca, spectral_cluster_input)
    elapsed_time = (time.time() - start_time)
    return pca_for_spectral[1],spectral_labels,elapsed_time

#plot the graph for  PCA Dimensionality Reduction with Spectral Clustering
# plot the clusters
def plot_Spectral_PCA(pca_components, Spectral_Labels, used_time):
    plt.scatter(pca_components[:, 0], pca_components[:, 1], c=Spectral_Labels, s=60)
    plt.title(f"PCA Dimension Reduction and Spectral clustering --- used_time :{used_time:.2f}s")
    plt.xlabel("PC1 ")
    plt.ylabel("PC2")
    plt.show()

#6. Spectral Clustering before LDA
def Spectral_LDA(n_lda,lda_classes,spectral_n_clusters,spectral_cluster_input):  # (LDA components, no.of clusters, lda input features, lda classes, kmeans input data)
    start_time = time.time()
    spectral_labels = SPECTRAL_CL(spectral_n_clusters, spectral_cluster_input)
    lda_out_for_spectral = LDA_DR(n_lda, spectral_cluster_input, lda_classes)
    elapsed_time = (time.time() - start_time)
    return lda_out_for_spectral[0],spectral_labels, elapsed_time


#plot the graph for LDA Dimensionality Reduction with Spectral Clustering
def plot_Spectral_LDA(x1_lda_spectral, x2_lda_spectral, Spectral_Labels, used_time):
    plt.scatter(x1_lda_spectral, x2_lda_spectral, c=Spectral_Labels, s=60)
    plt.title(f"LDA Dimension Reduction and Spectral clustering --- used_time :{used_time:.2f}s")
    plt.xlabel("LDA1 ")
    plt.ylabel("LDA2")
    plt.show()


#7. Spectral Clustering before t-SNE
def Spectral_tSNE(n_tsne,spectral_n_clusters,spectral_cluster_input):  # (TSNE embeddings, no.of clusters, TSNE input data, kmeans input data)
    start_time = time.time()
    spectral_labels = SPECTRAL_CL(spectral_n_clusters, spectral_cluster_input)
    tSNE_for_spectral = TSNE_DR(n_tsne, spectral_cluster_input)
    elapsed_time = (time.time() - start_time)
    return tSNE_for_spectral[0],spectral_labels, elapsed_time


# plot the clusters
def plot_Spectral_tSNE(x1_tsne_spectral, x2_tsne_spectral, Spectral_Labels, used_time):
    plt.scatter(x1_tsne_spectral, x2_tsne_spectral, c=Spectral_Labels, s=20)
    plt.title(f"tSNE Dimension Reduction after Spectral clustering--- used_time :{used_time:.2f}s")
    plt.xlabel("tSNE1 ")
    plt.ylabel("tSNE2")
    plt.show()

#8. Spectral Clustering before UMAP
def Spectral_UMAP(n_umap,spectral_n_clusters,spectral_cluster_input):  # (no of UMAP embeddings, no.of clusters, TSNE input data, kmeans input data)
    start_time = time.time()
    spectral_labels = SPECTRAL_CL(spectral_n_clusters, spectral_cluster_input)
    UMAP_for_spectral = UMAP_DR(n_umap, spectral_cluster_input)
    elapsed_time = (time.time() - start_time)
    return UMAP_for_spectral[0],spectral_labels,elapsed_time


# plot the clusters
def plot_Spectral_UMAP(x1_umap_spectral, x2_umap_spectral, Spectral_Labels, used_time):
    plt.scatter(x1_umap_spectral, x2_umap_spectral, c=Spectral_Labels, s=20)
    plt.title(f"UMAP Dimension Reduction after Spectral clustering--- used_time :{used_time:.2f}s")
    plt.xlabel("UMAP1 ")
    plt.ylabel("UMAP2")
    plt.show()