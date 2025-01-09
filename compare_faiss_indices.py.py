import os
import time
import faiss
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score
from tabulate import tabulate

def load_embeddings(file_path):
    """
    Load precomputed embeddings from a PyTorch file.

    Args:
        file_path (str): Path to the embeddings file.

    Returns:
        np.ndarray: Numpy array of embeddings.
        list: List of image filenames corresponding to the embeddings.
    """
    print(f"Loading embeddings from {file_path}...")
    train_embeddings = torch.load(file_path, weights_only=False)
    filenames = list(train_embeddings.keys())
    vectors = np.stack([v.numpy().squeeze() for v in train_embeddings.values()])
    print(f"Loaded {len(vectors)} embeddings with dimension {vectors.shape[1]}.")
    return vectors, filenames

def evaluate_search_accuracy(index, test_vectors, test_filenames, ground_truth_indices, k=5):
    """
    Evaluate the search accuracy of the index using Precision@k and Recall@k.

    Args:
        index (faiss.Index): The FAISS index to evaluate.
        test_vectors (np.ndarray): Embedding vectors for the test set.
        test_filenames (list): List of filenames for the test set.
        ground_truth_indices (dict): Ground truth mapping of test file to true neighbor indices.
        k (int): Number of nearest neighbors to retrieve.

    Returns:
        dict: Precision and Recall scores for the test set.
    """
    precision_list, recall_list = [], []

    for i, query_vector in enumerate(test_vectors):
        query_filename = test_filenames[i]
        distances, indices = index.search(query_vector.reshape(1, -1), k)

        retrieved_indices = set(indices[0])
        true_indices = set(ground_truth_indices[query_filename])

        # Calculate precision and recall
        tp = len(retrieved_indices & true_indices)  # True positives
        precision = tp / len(retrieved_indices) if retrieved_indices else 0
        recall = tp / len(true_indices) if true_indices else 0

        precision_list.append(precision)
        recall_list.append(recall)

    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)

    return {"precision": avg_precision, "recall": avg_recall}

def measure_query_latency(index, test_vectors, k=5):
    """
    Measure the average query latency of the index.

    Args:
        index (faiss.Index): The FAISS index to evaluate.
        test_vectors (np.ndarray): Embedding vectors for the test set.
        k (int): Number of nearest neighbors to retrieve.

    Returns:
        float: Average query latency in seconds.
    """
    latencies = []
    for query_vector in test_vectors:
        start_time = time.time()
        index.search(query_vector.reshape(1, -1), k)
        latencies.append(time.time() - start_time)

    avg_latency = np.mean(latencies)
    return avg_latency

def measure_memory_usage(index_path):
    """
    Measure the memory usage of the FAISS index file.

    Args:
        index_path (str): Path to the FAISS index file.

    Returns:
        float: Size of the index file in megabytes (MB).
    """
    file_size = os.path.getsize(index_path) / (1024 * 1024)  # Convert bytes to MB
    return file_size

def main():
    # Paths
    train_embeddings_path = "output/train_image_embeddings.pt"
    test_embeddings_path = "output/test_image_embeddings.pt"
    pq_index_path = "output/pq_index.faiss"
    hnsw_index_path = "output/hnsw_index.faiss"

    # Load embeddings
    train_embeddings, train_filenames = load_embeddings(train_embeddings_path)
    test_embeddings, test_filenames = load_embeddings(test_embeddings_path)

    # Load indices
    pq_index = faiss.read_index(pq_index_path)
    hnsw_index = faiss.read_index(hnsw_index_path)

    # Ground truth for evaluation (use brute-force nearest neighbors for ground truth)
    print("Generating ground truth with brute-force search...")
    ground_truth_index = faiss.IndexFlatL2(train_embeddings.shape[1])
    ground_truth_index.add(train_embeddings)
    ground_truth_indices = {}
    for i, test_vector in enumerate(test_embeddings):
        _, indices = ground_truth_index.search(test_vector.reshape(1, -1), k=5)
        ground_truth_indices[test_filenames[i]] = indices[0]

    # Evaluate PQ
    print("Evaluating PQ index...")
    pq_accuracy = evaluate_search_accuracy(pq_index, test_embeddings, test_filenames, ground_truth_indices, k=5)
    pq_latency = measure_query_latency(pq_index, test_embeddings, k=5)
    pq_memory = measure_memory_usage(pq_index_path)

    # Evaluate HNSW
    print("Evaluating HNSW index...")
    hnsw_accuracy = evaluate_search_accuracy(hnsw_index, test_embeddings, test_filenames, ground_truth_indices, k=5)
    hnsw_latency = measure_query_latency(hnsw_index, test_embeddings, k=5)
    hnsw_memory = measure_memory_usage(hnsw_index_path)

    # Prepare results for display
    results = [
        ["Product Quantization (PQ)", f"{pq_accuracy['precision']:.4f}", f"{pq_accuracy['recall']:.4f}", f"{pq_latency:.4f} s", f"{pq_memory:.2f} MB"],
        ["Hierarchical Navigable Small World (HNSW)", f"{hnsw_accuracy['precision']:.4f}", f"{hnsw_accuracy['recall']:.4f}", f"{hnsw_latency:.4f} s", f"{hnsw_memory:.2f} MB"]
    ]

    # Display results as a table
    headers = ["Method", "Precision@5", "Recall@5", "Avg Query Latency", "Index Size"]
    print(tabulate(results, headers=headers, tablefmt="pretty"))

if __name__ == "__main__":
    main()
