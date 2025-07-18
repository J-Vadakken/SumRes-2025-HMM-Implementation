import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
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
    
    def k_cluster_elbow(self, data_file_path, features, all_columns=False, scale=True, transition_flag=True):

        # Ensure directory exists
        plot_dir = "/home/jgv555/CS/ResSum2025/model/SumRes-2025-HMM-Implementation/plots/clustering"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)


        df = pd.read_csv(data_file_path)
        if (not(all_columns)):
            df = df[df["conv"] == True]


        df_features = df[["user_id", "date"]].copy()
        df_features['params'] = df['params'].apply(PlotObject.parse_array_string)
        if transition_flag:
            df_features['trans_g'] = df_features['params'].apply(lambda x: x[0])
            df_features['trans_b'] = df_features['params'].apply(lambda x: x[1])
        else: 
            df_features['trans_g'] = df_features['params'].apply(lambda x: 1/(1-x[0]))
            df_features['trans_b'] = df_features['params'].apply(lambda x: 1/(1-x[1]))
        df_features['prob_g'] = df_features['params'].apply(lambda x: x[2])
        df_features['prob_b'] = df_features['params'].apply(lambda x: x[3])
        df_features['init_prob_g'] = df_features['params'].apply(lambda x: x[4])
        df_features = df_features.dropna()



        if scale:
            scaler = MinMaxScaler()
            for feature in features:
                scaler.fit(df_features[[feature]])
                df_features[feature] = scaler.transform(df_features[[feature]])

        k_range = range(2, 20)
        sse = []
        for k in k_range:   
            km = KMeans(n_clusters=k)
            km.fit(df_features[features])
            sse.append(km.inertia_)

        plt.plot(k_range, sse, marker='o')
        plt.savefig("/home/jgv555/CS/ResSum2025/model/SumRes-2025-HMM-Implementation/plots/clustering/elbow_plot.png")

    def k_cluster(self, data_file_path, features, all_columns=False, n_clusters=6, scale=True, transition_flag=True):



        # Ensure directory exists
        plot_dir = "/home/jgv555/CS/ResSum2025/model/SumRes-2025-HMM-Implementation/plots/clustering"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)


        df = pd.read_csv(data_file_path)
        if (not(all_columns)):
            df = df[df["conv"] == True]

        df_features = df[["user_id", "date"]].copy()
        df_features['params'] = df['params'].apply(PlotObject.parse_array_string)
        if transition_flag:
            df_features['trans_g'] = df_features['params'].apply(lambda x: x[0])
            df_features['trans_b'] = df_features['params'].apply(lambda x: x[1])
        else: 
            df_features['trans_g'] = df_features['params'].apply(lambda x: 1/(1-x[0]))
            df_features['trans_b'] = df_features['params'].apply(lambda x: 1/(1-x[1]))

        df_features['prob_g'] = df_features['params'].apply(lambda x: x[2])
        df_features['prob_b'] = df_features['params'].apply(lambda x: x[3])
        df_features['init_prob_g'] = df_features['params'].apply(lambda x: x[4])
        df_features = df_features.dropna()

        if scale:
            scalers = {} # Store the scaler for inverse transform later
            for feature in features:
                scalers[feature] = MinMaxScaler()  # Create NEW scaler for each feature
                scalers[feature].fit(df_features[[feature]])
                df_features[feature] = scalers[feature].transform(df_features[[feature]])

        km = KMeans(n_clusters=n_clusters)
        y_predicted = km.fit_predict(df_features[features])
        
        scaled_centroids = km.cluster_centers_
        unscaled_centroids = scaled_centroids.copy()
        if scale:
            for i, feature in enumerate(features):
                unscaled_centroids[:, i] = scalers[feature].inverse_transform(scaled_centroids[:, i].reshape(-1, 1)).flatten()
        else:
            unscaled_centroids = scaled_centroids
        #2D scatter plot with PCA Dimensionality Reduction      
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(df_features[features])

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
        # plt.show()
        plt.savefig(f"/home/jgv555/CS/ResSum2025/model/SumRes-2025-HMM-Implementation/plots/clustering/kmeans_{n_clusters}_clusters.png")

        # 3D scatter plot with PCA Dimensionality Reduction
        pca_3d = PCA(n_components=3)
        data_3d = pca_3d.fit_transform(df_features[features])
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        for i in range(n_clusters):
            ax.scatter(data_3d[y_predicted == i, 0], data_3d[y_predicted == i, 1], 
                       data_3d[y_predicted == i, 2], 
                       color=colors[i % len(colors)], label=f'Cluster {i}')
        # Plot centroids
        centroids_3d = pca_3d.transform(km.cluster_centers_)
        ax.scatter(centroids_3d[:, 0], centroids_3d[:, 1], centroids_3d[:, 2], 
                   color='black', marker='X', s=200, label='Centroids')
        ax.set_xlabel(f"PC1 {pca_3d.explained_variance_ratio_[0]:.2f} variance")
        ax.set_ylabel(f"PC2 {pca_3d.explained_variance_ratio_[1]:.2f} variance")
        ax.set_zlabel(f"PC3 {pca_3d.explained_variance_ratio_[2]:.2f} variance")
        ax.set_title(f'3D K-Means Clustering with {n_clusters} Clusters')
        ax.legend()
        plt.savefig(f"/home/jgv555/CS/ResSum2025/model/SumRes-2025-HMM-Implementation/plots/clustering/kmeans_3d_{n_clusters}_clusters.png")
        # plt.show()
        plt.close()

        # Parallel Coordinates Plot
        df_features['cluster'] = y_predicted
        # Parallel coordinates plot
        plt.figure(figsize=(12, 6))
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']

        for cluster in range(n_clusters):
            cluster_data = df_features[df_features['cluster'] == cluster][features]
            for idx, row in cluster_data.iterrows():
                plt.plot(range(len(features)), row.values, color=colors[cluster % len(colors)], alpha=0.7)

        plt.xticks(range(len(features)), features, rotation=45)
        plt.ylabel('Scaled Feature Values')
        plt.title('Parallel Coordinates Plot of Clusters')
        plt.grid(True, alpha=0.3)
        plt.legend([f'Cluster {i}' for i in range(n_clusters)])
        plt.tight_layout()
        plt.savefig(f"/home/jgv555/CS/ResSum2025/model/SumRes-2025-HMM-Implementation/plots/clustering/parallel_coords_{n_clusters}_clusters.png")
        # plt.show()
        plt.close()

        plt.figure(figsize=(10, 6))
        centers_df = pd.DataFrame(km.cluster_centers_, 
                                columns=features,
                                index=[f'Cluster_{i}' for i in range(len(km.cluster_centers_))])

        sns.heatmap(centers_df, annot=True, cmap='viridis', fmt='.3f')
        plt.title('Cluster Centers Heatmap')
        plt.savefig(f"/home/jgv555/CS/ResSum2025/model/SumRes-2025-HMM-Implementation/plots/clustering/heatmap_{n_clusters}_clusters.png")
        # plt.show()
        plt.close()

        print(unscaled_centroids)
        # Print number of members and centroid values for each cluster
        cluster_sizes = pd.Series(y_predicted).value_counts().sort_index()
        print("\nCluster statistics:")
        for cluster_id, size in cluster_sizes.items():
            print(f"\nCluster {cluster_id}: {size} members")
            print("Centroid values:")
            for i, feature in enumerate(features):
                if not ((feature in ["trans_g", "trans_b"]) and (not transition_flag)):
                    print(f"  {feature}: {unscaled_centroids[cluster_id][i]:.4f}")
                if feature == "trans_g" and transition_flag:
                    num = 1/(1-unscaled_centroids[cluster_id][i])
                    print(f" {feature} Days in good state: {num:.4f}")
                if feature == "trans_b" and transition_flag:
                    num = 1/(1-unscaled_centroids[cluster_id][i])
                    print(f" {feature} Days in bad state: {num:.4f}")
                
                if feature == "trans_g" and not transition_flag:
                    num1 = unscaled_centroids[cluster_id][i]
                    num2 = 1 - 1/num1
                    print(f" {feature}: {num2:.4f}")
                    print(f" {feature} Days in good state: {num1:.4f}")
                if feature == "trans_b" and not transition_flag:
                    num1 = unscaled_centroids[cluster_id][i]
                    num2 = 1 - 1/num1
                    print(f" {feature}: {num2:.4f}")
                    print(f" {feature} Days in bad state: {num1:.4f}")




if __name__ == "__main__":
    plot_object = PlotObject()
    data_file_path = "/home/jgv555/CS/ResSum2025/model/SumRes-2025-HMM-Implementation/DataSummary/user_date_params_600.csv"
    print("PlotObject instance created successfully.")

    features = ['trans_g', 'trans_b', 'prob_g', 'prob_b', 'init_prob_g']

    plot_object.k_cluster_elbow(data_file_path, features, all_columns=False, scale=True, transition_flag=False) #all_columns=False)
    plot_object.k_cluster(data_file_path, features, all_columns=False, n_clusters=3, scale=True, transition_flag=False) #all_columns=False, n_clusters=6)
    plot_object.k_cluster(data_file_path, features, all_columns=False, n_clusters=4, scale=True, transition_flag=False) #all_columns=False, n_clusters=6)
    