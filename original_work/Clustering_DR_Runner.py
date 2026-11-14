# Runner
import numpy as np
import os
import torch

#Import functions form the library
#for clustering
from Clustering_DR_Library import KMEANS_CL, SPECTRAL_CL

#for Dimensionality Reduction
from Clustering_DR_Library import PCA_DR, LDA_DR,TSNE_DR,UMAP_DR
from Clustering_DR_Library import plot_PCA, plot_LDA, plot_TSNE,plot_UMAP

#for Dimensionality Reduction before Clustering
from Clustering_DR_Library import PCA_kmeans,LDA_kmeans,tSNE_kmeans,UMAP_kmeans
from Clustering_DR_Library import plot_PCA_kmeans,plot_LDA_kmeans,plot_tSNE_kmeans,plot_UMAP_kmeans
from Clustering_DR_Library import PCA_Spectral,LDA_Spectral,tSNE_Spectral,UMAP_Spectral
from Clustering_DR_Library import plot_PCA_Spectral,plot_LDA_Spectral, plot_tSNE_Spectral, plot_UMAP_Spectral

#for Dimensionality Reduction after Clustering
from Clustering_DR_Library import kmeans_PCA,kmeans_LDA,kmeans_tSNE,kmeans_UMAP
from Clustering_DR_Library import plot_kmeans_PCA,plot_kmeans_LDA,plot_kmeans_tSNE,plot_kmeans_UMAP
from Clustering_DR_Library import Spectral_PCA,Spectral_LDA,Spectral_tSNE,Spectral_UMAP
from Clustering_DR_Library import plot_Spectral_PCA,plot_Spectral_LDA,plot_Spectral_tSNE,plot_Spectral_UMAP

#get the files from the fi,e path
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

#give the file path
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
# get user input for the main method
get_user_function = input(
    "Select the number (from 1,2,3) for the function from the following list: \n 1. Clustering \n 2. Dimensionality Reduction \n 3. Clustering after Dimensionality Reduction \n 4. Clustering before Dimensionality Reduction\n: ")

#Perform clustering
if get_user_function == "1":
    print("You selected Clustering")
    get_clustering_method = input(
        "Select the method for the clustering\n 1.KMeans Clustering \n 2.Spectral Clustering \n:")
    # perform kmeans clustering
    if get_clustering_method == "1":
        print("You selected KMeans clustering, This method does not have a visualization")
        get_n_clusters = input("Enter the number of clusters\n: ")
        k_clusters = int(get_n_clusters)
        k_cluster_input = feature_array
        KMEANS_CL(k_clusters, k_cluster_input)
    #perform spectral clustering
    elif get_clustering_method == "2":
        print("You selected Spectral clustering, This method does not have a visualization")
        get_n_clusters = input("Enter the number of clusters\n: ")
        spectral_n_clusters = int(get_n_clusters)
        spectral_cluster_input = feature_array
        SPECTRAL_CL(spectral_n_clusters, spectral_cluster_input)

#Perform dimensionality reduction
elif get_user_function == "2":
    print("You selected Dimensionality Reduction")
    get_DR_method = input(
        "Select the method for the dimensionality Reduction \n 1.PCA \n 2.LDA \n 3.t-SNE \n 4.UMAP \n:")

    # 1.PCA
    if get_DR_method == "1":
        print("You selected PCA dimensionality reduction technique")
        get_dimensions = input("Enter the number of dimensions(Only 2 available): \n")
        n_pca = int(get_dimensions)
        pca_input = feature_array

        PCA_out = PCA_DR(n_pca, pca_input)
        pca_variance_ratio = PCA_out[0]
        pca_out_components = PCA_out[1]
        used_time = PCA_out[2]
        plot_PCA(pca_out_components, pca_variance_ratio, used_time)
        
    # 2.LDA
    elif get_DR_method == "2":
        print("You selected LDA dimensionality reduction technique")
        get_dimensions = input("Enter the number of dimensions(Only 2 available): \n")
        n_lda = int(get_dimensions)
        lda_data = feature_array
        lda_classes = classes

        LDA_out = LDA_DR(n_lda, lda_data, lda_classes)
        lda_out_components = LDA_out[0]
        used_time = LDA_out[1]

        x1 = lda_out_components[:, 0]
        x2 = lda_out_components[:, 1]
        plot_LDA(x1, x2, used_time)

    # 3.t-SNE
    elif get_DR_method == "3":
        print("You selected t-SNE dimensionality reduction technique")
        get_dimensions = input("Enter the number of dimensions(Only 2 available): \n")
        n_tsne = int(get_dimensions)
        tsne_input = feature_array

        tsne_out = TSNE_DR(n_tsne, tsne_input)
        tsne_components = tsne_out[0]
        used_time = tsne_out[1]
        x1 = tsne_components[:, 0]
        x2 = tsne_components[:, 1]
        plot_TSNE(x1, x2, used_time)

    # 4.UMAP
    elif get_DR_method == "4":
        print("You selected UMAP dimensionality reduction technique")
        get_dimensions = input("Enter the number of dimensions(Only 2 available): \n")
        n_umap = int(get_dimensions)
        umap_input = feature_array

        umap_out = UMAP_DR(n_umap, umap_input)
        umap_embedding = umap_out[0]
        used_time = umap_out[1]

        x1 = umap_embedding[:, 0]
        x2 = umap_embedding[:, 1]
        plot_UMAP(x1, x2, used_time)

#Perform dimensionality reduction before Clustering
elif get_user_function == "3":
    # 1. PCA with KMeans
    get_DR_Clustering_method = input("Select the method for the dimensionality Reduction \n 1.PCA and KMeans \n 2.PCA and Spectral \n 3.LDA and KMeans \n 4.LDA and Spectral  \n 5.t-SNE and KMeans\n 6.t-SNE and Spectral \n 7.UMAP and KMeans \n 8.UMAP and Spectral \n:")
    if get_DR_Clustering_method == "1":
        get_dimensions = input("Enter the number of dimensions(Only 2 available): \n")
        get_n_clusters = input("Enter the number of clusters\n: ")

        n_pca = int(get_dimensions)
        k_clusters = int(get_n_clusters)
        pca_input = feature_array
        kmeans_pca_output = PCA_kmeans(n_pca, k_clusters, pca_input)  # cluster input = DR output

        kmeans_after_PCA = kmeans_pca_output[0]
        pca_components_kmeans = kmeans_pca_output[1]
        used_time: float = kmeans_pca_output[2]

        kmeans_cluster_labels_after_PCA = kmeans_after_PCA[0]
        kmeans_centroids_after_PCA = kmeans_after_PCA[1]
        plot_PCA_kmeans(pca_components_kmeans, kmeans_cluster_labels_after_PCA, kmeans_centroids_after_PCA, used_time)

    # 2. PCA with Spectral
    elif get_DR_Clustering_method == "2":
        print("You selected PCA and Spectral")
        get_dimensions = input("Enter the number of dimensions(Only 2 available): \n")
        get_n_clusters = input("Enter the number of clusters\n: ")

        n_pca = int(get_dimensions)
        spectral_n_clusters = int(get_n_clusters)
        pca_input = feature_array

        # Get the kmeans labels, centroids and pca components after the dimensionality reduction and clustering
        spectral_pca_output = PCA_Spectral(n_pca, spectral_n_clusters, pca_input)  # cluster input = DR output
        spectral_cluster_labels_after_PCA = spectral_pca_output[0]
        pca_components_spectral = spectral_pca_output[1]
        used_time = spectral_pca_output[2]

        plot_PCA_Spectral(pca_components_spectral, spectral_cluster_labels_after_PCA, used_time)

    # 3. LDA with KMeans
    elif get_DR_Clustering_method == "3":
        print("You selected LDA and KMeans")
        get_dimensions = input("Enter the number of dimensions(Only 2 available): \n")
        get_n_clusters = input("Enter the number of clusters\n: ")

        n_lda = int(get_dimensions)
        k_clusters = int(get_n_clusters)

        lda_data = feature_array
        lda_classes = classes

        # take the output for plotting
        kmeans_LDA_output = LDA_kmeans(n_lda, lda_data, lda_classes, k_clusters)

        kmeans_after_LDA = kmeans_LDA_output[0]
        lda_components_kmeans = kmeans_LDA_output[1]
        used_time = kmeans_LDA_output[2]
        kmeans_cluster_labels_after_LDA = kmeans_after_LDA[0]
        kmeans_centroids_after_LDA = kmeans_after_LDA[1]

        x1_lda_kmeans = lda_components_kmeans[:, 0]
        x2_lda_kmeans = lda_components_kmeans[:, 1]
        plot_LDA_kmeans(x1_lda_kmeans, x2_lda_kmeans, kmeans_cluster_labels_after_LDA, kmeans_centroids_after_LDA,used_time)

    # 4. LDA with Spectral
    elif get_DR_Clustering_method == "4":
        print("You selected LDA and Spectral")
        get_dimensions = input("Enter the number of dimensions(Only 2 available): \n")
        get_n_clusters = input("Enter the number of clusters\n: ")

        n_lda = int(get_dimensions)
        spectral_n_clusters = int(get_n_clusters)
        lda_data = feature_array
        lda_classes = classes

        # take the output for plotting
        spectral_LDA_output = LDA_Spectral(n_lda, lda_data, lda_classes, spectral_n_clusters)
        spectral_cluster_labels_after_LDA = spectral_LDA_output[0]
        lda_components_spectral = spectral_LDA_output[1]
        used_time = spectral_LDA_output[2]

        # lda components for plotting
        x1_lda_spectral = lda_components_spectral[:, 0]
        x2_lda_spectral = lda_components_spectral[:, 1]
        plot_LDA_Spectral(x1_lda_spectral, x2_lda_spectral, spectral_cluster_labels_after_LDA, used_time)

    # 5. t-SNE with KMeans
    elif get_DR_Clustering_method == "5":
        print("You selected t-SNE and KMeans")
        get_dimensions = input("Enter the number of dimensions(Only 2 available): \n")
        get_n_clusters = input("Enter the number of clusters\n: ")

        n_tsne = int(get_dimensions)
        k_clusters = int(get_n_clusters)
        tsne_input = feature_array
        # take the output for plotting
        kmeans_tSNE_output = tSNE_kmeans(n_tsne, tsne_input, k_clusters)

        # output from the tSNE
        kmeans_after_tSNE = kmeans_tSNE_output[0]
        tSNE_components_kmeans = kmeans_tSNE_output[1]
        used_time = kmeans_tSNE_output[2]
        # tSNE components for plotting
        x1_tsne_kmeans = tSNE_components_kmeans[:, 0]
        x2_tsne_kmeans = tSNE_components_kmeans[:, 1]

        # outputs from the kmeans for plotting
        kmeans_cluster_labels_after_tSNE = kmeans_after_tSNE[0]
        kmeans_centroids_after_tSNE = kmeans_after_tSNE[1]

        plot_tSNE_kmeans(x1_tsne_kmeans, x2_tsne_kmeans, kmeans_cluster_labels_after_tSNE, kmeans_centroids_after_tSNE,
                         used_time)

    # 6. t-SNE with Spectral
    elif get_DR_Clustering_method == "6":
        print("You selected t-SNE and Spectral")
        get_dimensions = input("Enter the number of dimensions(Only 2 available): \n")
        get_n_clusters = input("Enter the number of clusters\n: ")

        n_tsne = int(get_dimensions)
        spectral_n_clusters = int(get_n_clusters)
        tsne_input = feature_array

        spectral_tSNE_output = tSNE_Spectral(n_tsne, tsne_input, spectral_n_clusters)
        spectral_cluster_labels_after_tSNE = spectral_tSNE_output[0]
        tSNE_components_spectral = spectral_tSNE_output[1]
        used_time = spectral_tSNE_output[2]
        # output from the tSNE

        # tSNE components for plotting
        x1_tsne_spectral = tSNE_components_spectral[:, 0]
        x2_tsne_spectral = tSNE_components_spectral[:, 1]
        plot_tSNE_Spectral(x1_tsne_spectral, x2_tsne_spectral, spectral_cluster_labels_after_tSNE, used_time)

    # 7.UMAP with KMeans
    elif get_DR_Clustering_method == "7":
        print("You selected UMAP and KMeans")
        get_dimensions = input("Enter the number of dimensions(Only 2 available): \n")
        get_n_clusters = input("Enter the number of clusters\n: ")

        n_umap = int(get_dimensions)
        k_clusters = int(get_n_clusters)
        umap_input = feature_array
        kmeans_UMAP_output = UMAP_kmeans(n_umap, umap_input, k_clusters)

        # output from the tSNE
        kmeans_after_UMAP = kmeans_UMAP_output[0]
        UMAP_components_kmeans = kmeans_UMAP_output[1]
        used_time = kmeans_UMAP_output[2]
        # tSNE components for plotting
        x1_umap_kmeans = UMAP_components_kmeans[:, 0]
        x2_umap_kmeans = UMAP_components_kmeans[:, 1]

        # outputs from the kmeans for plotting
        kmeans_cluster_labels_after_umap = kmeans_after_UMAP[0]
        kmeans_centroids_after_umap = kmeans_after_UMAP[1]
        plot_UMAP_kmeans(x1_umap_kmeans, x2_umap_kmeans, kmeans_cluster_labels_after_umap, kmeans_centroids_after_umap,
                         used_time)

    # 8. UMAP with Spectral
    elif get_DR_Clustering_method == "8":
        print("You selected UMAP and Spectral")
        get_dimensions = input("Enter the number of dimensions(Only 2 available): \n")
        get_n_clusters = input("Enter the number of clusters\n: ")

        n_umap = int(get_dimensions)
        spectral_n_clusters = int(get_n_clusters)
        umap_input = feature_array

        # take the output for plotting
        spectral_UMAP_output = UMAP_Spectral(n_umap, umap_input, spectral_n_clusters)
        spectral_cluster_labels_after_umap = spectral_UMAP_output[0]
        UMAP_components_spectral = spectral_UMAP_output[1]
        used_time = spectral_UMAP_output[2]

        # tSNE components for plotting
        x1_umap_spectral = UMAP_components_spectral[:, 0]
        x2_umap_spectral = UMAP_components_spectral[:, 1]
        plot_UMAP_Spectral(x1_umap_spectral, x2_umap_spectral, spectral_cluster_labels_after_umap, used_time)


#Perform dimensionality reduction after Clustering
elif get_user_function == "4":

    # 1. PCA with KMeans
    get_clustering_DR_method = input("Select the method for the dimensionality Reduction \n 1.KMeans and PCA \n 2.Spectral and PCA \n 3.KMeans and LDA \n 4.Spectral and LDA  \n 5.KMeans and t-SNE\n 6.Spectral and t-SNE \n 7.KMeans and UMAP \n 8.Spectral and UMAP \n:")
    if get_clustering_DR_method == "1":
        print("You selected to do KMeans clustering first and then PCA dimensionality reduction second ")
        get_dimensions = input("Enter the number of dimensions(Only 2 available): \n")
        get_n_clusters = input("Enter the number of clusters\n: ")

        k_clusters = int(get_n_clusters)
        n_pca = int(get_dimensions)
        k_cluster_input = feature_array
        kmeans_pca_output = kmeans_PCA(n_pca, k_clusters, k_cluster_input)  # cluster input = DR output

        PCA_Components = kmeans_pca_output[0]
        K_Labels = kmeans_pca_output[1]
        used_time: float = kmeans_pca_output[2]
        plot_kmeans_PCA(PCA_Components,K_Labels,used_time)

    # 2. PCA with Spectral
    elif get_clustering_DR_method == "2":
        print("You selected PCA and Spectral")
        get_dimensions = input("Enter the number of dimensions(Only 2 available): \n")
        get_n_clusters = input("Enter the number of clusters\n: ")

        n_pca = int(get_dimensions)
        spectral_cluster_input = feature_array
        spectral_n_clusters = int(get_n_clusters)

        # Get the kmeans labels, centroids and pca components after the dimensionality reduction and clustering
        spectral_pca_output = Spectral_PCA(n_pca, spectral_n_clusters,spectral_cluster_input)  # cluster input = DR output

        pca_components = spectral_pca_output[0]
        Spectral_Labels = spectral_pca_output[1]
        used_time = spectral_pca_output[2]

        plot_Spectral_PCA(pca_components, Spectral_Labels, used_time)

    # 3. LDA with KMeans
    if get_clustering_DR_method == "3":
        print("You selected to do KMeans clustering first and then PCA dimensionality reduction second ")
        get_dimensions = input("Enter the number of dimensions(Only 2 available): \n")
        get_n_clusters = input("Enter the number of clusters\n: ")

        n_lda = int(get_dimensions)
        k_clusters = int(get_n_clusters)
        k_cluster_input = feature_array
        lda_classes = classes

        # take the output for plotting
        kmeans_LDA_output = kmeans_LDA(n_lda, k_cluster_input, lda_classes, k_clusters)

        lda_components = kmeans_LDA_output[0]
        K_Labels = kmeans_LDA_output[1]
        used_time = kmeans_LDA_output[2]

        x1_lda_kmeans = lda_components[:, 0]
        x2_lda_kmeans = lda_components[:, 1]
        plot_kmeans_LDA(x1_lda_kmeans, x2_lda_kmeans, K_Labels, used_time)

    # 4. LDA with Spectral
    if get_clustering_DR_method == "4":
        print("You selected to do Spectral clustering first and then LDA dimensionality reduction second ")
        get_dimensions = input("Enter the number of dimensions(Only 2 available): \n")
        get_n_clusters = input("Enter the number of clusters\n: ")

        n_lda = int(get_dimensions)
        lda_classes = classes
        spectral_cluster_input = feature_array
        spectral_n_clusters = int(get_n_clusters)
        # take the output for plotting
        spectral_LDA_output = Spectral_LDA(n_lda,lda_classes,spectral_n_clusters,spectral_cluster_input)

        lda_components = spectral_LDA_output[0]
        Spectral_Labels = spectral_LDA_output[1]
        used_time = spectral_LDA_output[2]

        x1_lda_kmeans = lda_components[:, 0]
        x2_lda_kmeans = lda_components[:, 1]
        plot_Spectral_LDA(x1_lda_kmeans, x2_lda_kmeans, Spectral_Labels, used_time)

    # 5. t-SNE with KMeans
    elif get_clustering_DR_method == "5":
        print("You selected t-SNE and KMeans")
        get_dimensions = input("Enter the number of dimensions(Only 2 available): \n")
        get_n_clusters = input("Enter the number of clusters\n: ")

        n_tsne = int(get_dimensions)
        k_clusters = int(get_n_clusters)
        k_cluster_input = feature_array
        # take the output for plotting
        kmeans_tSNE_output = kmeans_tSNE(n_tsne,k_clusters,k_cluster_input)

        # output from the tSNE
        tSNE_components = kmeans_tSNE_output[0]
        K_Labels = kmeans_tSNE_output[1]
        used_time = kmeans_tSNE_output[2]
        # tSNE components for plotting
        x1_tsne_kmeans = tSNE_components[:, 0]
        x2_tsne_kmeans = tSNE_components[:, 1]

        plot_kmeans_tSNE(x1_tsne_kmeans, x2_tsne_kmeans, K_Labels, used_time)

    #6. tSNE with Spectral
    elif get_clustering_DR_method == "6":
        print("You selected t-SNE and Spectral")
        get_dimensions = input("Enter the number of dimensions(Only 2 available): \n")
        get_n_clusters = input("Enter the number of clusters\n: ")

        n_tsne = int(get_dimensions)
        spectral_cluster_input = feature_array
        spectral_n_clusters = int(get_n_clusters)
        Spectral_tSNE_output = Spectral_tSNE(n_tsne, spectral_n_clusters, spectral_cluster_input)

        # output from the tSNE
        tSNE_components = Spectral_tSNE_output[0]
        Spectral_Labels = Spectral_tSNE_output[1]
        used_time = Spectral_tSNE_output[2]
        # tSNE components for plotting
        x1_tsne_kmeans = tSNE_components[:, 0]
        x2_tsne_kmeans = tSNE_components[:, 1]
        plot_Spectral_tSNE(x1_tsne_kmeans, x2_tsne_kmeans, Spectral_Labels, used_time)

    # 7. UMAP with KMeans
    elif get_clustering_DR_method == "7":
        print("You selected UMAP and KMeans")
        get_dimensions = input("Enter the number of dimensions(Only 2 available): \n")
        get_n_clusters = input("Enter the number of clusters\n: ")

        n_umap = int(get_dimensions)
        k_cluster_input = feature_array
        k_clusters = int(get_n_clusters)
        kmeans_UMAP_output = kmeans_UMAP(n_umap,k_clusters,k_cluster_input,)

        # output from the tSNE
        UMAP_components = kmeans_UMAP_output[0]
        K_Labels = kmeans_UMAP_output[1]
        used_time = kmeans_UMAP_output[2]

        # tSNE components for plotting
        x1_umap_kmeans = UMAP_components[:, 0]
        x2_umap_kmeans = UMAP_components[:, 1]
        plot_kmeans_UMAP(x1_umap_kmeans, x2_umap_kmeans, K_Labels, used_time)

    # 7.UMAP with KMeans

    elif get_clustering_DR_method == "8":
        print("You selected UMAP and KMeans")
        get_dimensions = input("Enter the number of dimensions(Only 2 available): \n")
        get_n_clusters = input("Enter the number of clusters\n: ")

        n_umap = int(get_dimensions)
        spectral_cluster_input = feature_array
        spectral_n_clusters = int(get_n_clusters)
        Spectral_UMAP_output = Spectral_UMAP(n_umap, spectral_n_clusters, spectral_cluster_input)

        # output from the tSNE
        UMAP_components = Spectral_UMAP_output[0]
        Spectral_Labels = Spectral_UMAP_output[1]
        used_time = Spectral_UMAP_output[2]

        # tSNE components for plotting
        x1_umap_kmeans = UMAP_components[:, 0]
        x2_umap_kmeans = UMAP_components[:, 1]
        plot_Spectral_UMAP(x1_umap_kmeans, x2_umap_kmeans, Spectral_Labels, used_time)