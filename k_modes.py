from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import random
from kmodes.kmodes import KModes
from sklearn.preprocessing import LabelEncoder

class ClusteringModel:
    def __init__(self, file_names, file_path):
        self.file_names = file_names
        self.file_path = file_path
        self.dfs = []

    def load_and_clean_data(self, file_name):
        try:
            # Read a random sample of 5% of the file
            df = pd.read_csv(self.file_path + file_name, header=0, skiprows=lambda i: i > 0 and random.random() > 0.05)
            self.dfs.append(df)
            print(f"Loaded {file_name} successfully. Shape: {df.shape}")
        except pd.errors.ParserError as e:
            print(f"Error reading file {file_name}: {e}")

    def merge_datasets(self):
        if not self.dfs:
            print("No valid DataFrames loaded. Exiting.")
            exit()
        merged_df = self.dfs[0].merge(self.dfs[1], on='name', how='inner').merge(self.dfs[2], on='name', how='inner')
        return merged_df

    def encode_categorical_variables(self, df):
        label_encoder = LabelEncoder()
        df['gender'] = label_encoder.fit_transform(df['gender'])
        df['level3_main_occ'] = label_encoder.fit_transform(df['level3_main_occ'])
        df['level2_second_occ'] = label_encoder.fit_transform(df['level2_second_occ'])
        return df

    def fit_kmodes_model(self, df, features, k):
        kmodes = KModes(n_clusters=k, init='Cao', verbose=1)
        clusters = kmodes.fit_predict(df[features])
        df['kmodes_cluster'] = clusters
        name_kmodes_mapping = df[['name', 'kmodes_cluster']].set_index('name')['kmodes_cluster'].to_dict()
        return df, name_kmodes_mapping

    def get_names_for_profession(self, df, profession):
        selected_cluster = df[df['level3_main_occ'] == LabelEncoder.transform([profession])[0]]['kmodes_cluster'].unique()
        if len(selected_cluster) > 0:
            return df[df['kmodes_cluster'] == selected_cluster[0]]['name'].tolist()
        else:
            return []

    def fit_kmeans_model(self, df, features, k):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df[features])
        df['kmeans_cluster'] = kmeans.labels_
        name_kmeans_mapping = df[['name', 'kmeans_cluster']].set_index('name')['kmeans_cluster'].to_dict()
        return df, name_kmeans_mapping

    def get_similar_names_kmeans(self, df, selected_name):
        selected_cluster = df[df['name'] == selected_name]['kmeans_cluster'].values[0]
        similar_names = df[df['kmeans_cluster'] == selected_cluster]['name'].tolist()
        return similar_names


# Example usage:
file_names = [
    'cross-verified-filtered_2023-11-18.csv',
    'names_by_state_cleaned.csv',
    'names_by_yob_cleaned.csv'
]
file_path = '../datasets/cleaned_datasets/'

clustering_model = ClusteringModel(file_names, file_path)

# Load and clean each dataset
for file_name in clustering_model.file_names:
    clustering_model.load_and_clean_data(file_name)

# Merge datasets
merged_df = clustering_model.merge_datasets()

# Encode categorical variables
encoded_df = clustering_model.encode_categorical_variables(merged_df)

# Fit K-Modes model
features_to_cluster_kmodes = ['gender', 'level3_main_occ']
k_modes_clusters = 5
final_df_kmodes, name_kmodes_mapping = clustering_model.fit_kmodes_model(encoded_df, features_to_cluster_kmodes, k_modes_clusters)

# Example: Get names for the 'actor' profession using K-Modes
actor_names_kmodes = clustering_model.get_names_for_profession(final_df_kmodes, 'actor')
print("Names associated with the 'actor' profession (K-Modes):")
print(actor_names_kmodes)
