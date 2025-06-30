import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import ast



class PlotObject:
    def __init__(self):
        return None
    
    def parse_array_string(x):
        # Remove quotes and split on whitespace
        try:
            # Clean the string and convert to array
            clean_str = x.strip('"[]').replace('\n', ' ')
            numbers = np.fromstring(clean_str, sep=' ')
            return numbers  # Return first element
        except:
            return None
    
    def k_cluster_elbow(self, data_file_path, all_columns=False, scale=True):
        df = pd.read_csv(data_file_path)
        if (not(all_columns)):
            df = df[df["conv"] == True]


        df_features = df[["user_id", "date"]].copy()
        df_features['params'] = df['params'].apply(PlotObject.parse_array_string)
        df_features['trans_g'] = df_features['params'].apply(lambda x: x[0])
        df_features['trans_b'] = df_features['params'].apply(lambda x: x[1])
        df_features['prob_g'] = df_features['params'].apply(lambda x: x[2])
        df_features['prob_b'] = df_features['params'].apply(lambda x: x[3])
        df_features['init_prob_g'] = df_features['params'].apply(lambda x: x[4])
        df_features = df_features.dropna()



        if scale:
            scaler = MinMaxScaler()
            for feature in ['trans_g', 'trans_b', 'prob_g', 'prob_b', 'init_prob_g']:
                scaler.fit(df_features[[feature]])
                df_features[feature] = scaler.transform(df_features[[feature]])

        k_range = range(2, 20)
        sse = []
        for k in k_range:   
            km = KMeans(n_clusters=k)
            km.fit(df_features[['trans_g', 'trans_b', 'prob_g', 'prob_b', 'init_prob_g']])
            sse.append(km.inertia_)

        plt.plot(k_range, sse, marker='o')
        plt.savefig("/home/jgv555/CS/ResSum2025/model/SumRes-2025-HMM-Implementation/plots/clustering/elbow_plot.png")

    def k_cluster(self, data_file_path, all_columns=False, n_clusters=6, scale=True):
        df = pd.read_csv(data_file_path)
        if (not(all_columns)):
            df = df[df["conv"] == True]

        df_features = df[["user_id", "date"]].copy()
        df_features['params'] = df['params'].apply(PlotObject.parse_array_string)
        df_features['trans_g'] = df_features['params'].apply(lambda x: x[0])
        df_features['trans_b'] = df_features['params'].apply(lambda x: x[1])
        df_features['prob_g'] = df_features['params'].apply(lambda x: x[2])
        df_features['prob_b'] = df_features['params'].apply(lambda x: x[3])
        df_features['init_prob_g'] = df_features['params'].apply(lambda x: x[4])
        df_features = df_features.dropna()

        
        if scale: 
            scaler = MinMaxScaler()
            for feature in ['trans_g', 'trans_b', 'prob_g', 'prob_b', 'init_prob_g']:
                scaler.fit(df_features[[feature]])
                df_features[feature] = scaler.transform(df_features[[feature]])

        km = KMeans(n_clusters=n_clusters)
        y_predicted = km.fit_predict(df_features[['trans_g', 'trans_b', 'prob_g', 'prob_b', 'init_prob_g']])
        
        #2D scatter plot with PCA Dimensionality Reduction      
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(df_features[['trans_g', 'trans_b', 'prob_g', 'prob_b', 'init_prob_g']])

        # Plot clusters
        plt.figure(figsize=(10, 6))
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
        for i in range(n_clusters):
            plt.scatter(data_2d[y_predicted == i, 0], data_2d[y_predicted == i, 1], 
                        color=colors[i % len(colors)], label=f'Cluster {i}')
        # Plot centroids
        centroids = pca.transform(km.cluster_centers_)
        plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='X', s=200, label='Centroids')

        plt.xlabel(f"PC1 {pca.explained_variance_ratio_[0]:.2f} variance")
        plt.ylabel(f"PC2 {pca.explained_variance_ratio_[1]:.2f} variance")
        plt.title(f'K-Means Clustering with {n_clusters} Clusters')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        plt.savefig(f"/home/jgv555/CS/ResSum2025/model/SumRes-2025-HMM-Implementation/plots/clustering/kmeans_{n_clusters}_clusters.png")





if __name__ == "__main__":
    plot_object = PlotObject()
    data_file_path = "/home/jgv555/CS/ResSum2025/model/SumRes-2025-HMM-Implementation/DataSummary/user_date_params.csv"
    print("PlotObject instance created successfully.")
    # You can call methods on plot_object here, e.g., plot_object.k_cluster('data.csv')
    plot_object.k_cluster_elbow(data_file_path, all_columns=False, scale=True) #all_columns=False)
    plot_object.k_cluster(data_file_path, all_columns=False, n_clusters=3, scale=True) #all_columns=False, n_clusters=6)