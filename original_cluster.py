import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import plotly.express as px

class NameProfessionAnalyzer:
    def __init__(self, dataset_path):
        self.df = pd.read_csv(dataset_path)
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.clusters = None

    def preprocess_data(self):
        self.df['text'] = self.df['name']
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['text'])

    def perform_clustering(self, eps=0.3, min_samples=2):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.clusters = dbscan.fit_predict(self.tfidf_matrix)

    def find_top_professions(self, name, gender):
        query = self.tfidf_vectorizer.transform([name])
        query_cluster = self.clusters[-1]
        same_cluster_indices = [i for i, cluster_id in enumerate(self.clusters) if cluster_id == query_cluster]
        same_cluster_data = self.df.iloc[same_cluster_indices]

        filtered_data = same_cluster_data[(same_cluster_data['gender'] == gender)]
        top_professions = filtered_data['level3_main_occ'].value_counts().head(5)
        return top_professions

    def display_top_professions(self, top_professions):
        fig = px.bar(top_professions, x=top_professions.index, y=top_professions.values, title="Top Professions")
        fig.show()

    def analyze_data(self):
        self.preprocess_data()
        self.perform_clustering()
        user_input = input("Enter 'name' or 'profession': ")
        if user_input == "name":
            name = input("Enter a name: ")
            gender = input("Enter gender: ")
            top_professions = self.find_top_professions(name, gender)
            self.display_top_professions(top_professions)
        elif user_input == "profession":
            profession = input("Enter a profession: ")
            gender = input("Enter gender: ")
            # Implement the logic to find common names for the given profession
            # Display the results using Plotly or other visualization libraries
        else:
            print("Invalid input. Please enter 'name' or 'profession'.")

if __name__ == "__main__":
    dataset_path = '/Users/vincentcholewa/Documents/GAT/DVA/Project/datasets/cross-verified-filtered.csv'
    analyzer = NameProfessionAnalyzer(dataset_path)
    analyzer.analyze_data()
