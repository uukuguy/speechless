import json
import numpy as np
import argparse
from sklearn.cluster import KMeans


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--embedding_path', type=str, help="save/alpaca_gpt4/embeddings/52002.npy")
    argparse.add_argument('--instruction_path', type=str, help="datasets/alpaca_gpt4/alpaca_gpt4_data.json")
    argparse.add_argument('--save_path', type=str, help="datasets/alpaca_gpt4/alpaca_gpt4_kmeans_100.json")
    args = argparse.parse_args()

    EMBEDDING_PATH = args.embedding_path
    INSTRUCTION_PATH = args.instruction_path
    SAVE_PATH = args.save_path

    embeddings = []
    embeddings.extend(np.load(f'{EMBEDDING_PATH}'))

    print(len(embeddings))
    print("K-MEANS")

    # KMeans clustering
    kmeans = KMeans(n_clusters=100, random_state=0).fit(embeddings)

    # kmeans.cluster_centers_
    def find_nearest(embedding, embeddings):
        distances = ((embeddings - embedding) ** 2).sum(axis=1)
        return distances.argmin()
    cluster_center_indices = [find_nearest(center, embeddings) for center in kmeans.cluster_centers_]

    print(cluster_center_indices)

    with open(f"{INSTRUCTION_PATH}", "r") as f:
        data = json.load(f)
    kmeans_sample = [data[i] for i in cluster_center_indices]

    kmeans_sample = json.dumps(kmeans_sample, indent=4)
    with open(f'{SAVE_PATH}', 'w') as f:
        f.write(kmeans_sample)
