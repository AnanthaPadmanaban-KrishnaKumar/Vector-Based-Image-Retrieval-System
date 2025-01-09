import os
import faiss
import torch
import numpy as np

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

def create_hnsw_index(vectors, ef_construction=200, M=32):
    """
    Create a Hierarchical Navigable Small World (HNSW) index for the given embeddings.

    Args:
        vectors (np.ndarray): Embedding vectors.
        ef_construction (int): Number of neighbors considered during construction.
        M (int): Number of bi-directional links per node in the graph.

    Returns:
        faiss.IndexHNSWFlat: Trained HNSW index.
    """
    d = vectors.shape[1]  # Dimension of the embeddings
    print(f"Creating an HNSW index with ef_construction={ef_construction} and M={M}...")

    # Define HNSW index
    hnsw_index = faiss.IndexHNSWFlat(d, M)

    # Set the ef_construction parameter for building the graph
    hnsw_index.hnsw.efConstruction = ef_construction

    # Add embeddings to the HNSW index
    print("Adding embeddings to the HNSW index...")
    hnsw_index.add(vectors)
    print(f"Added {hnsw_index.ntotal} embeddings to the index.")

    return hnsw_index

def save_index(index, output_path):
    """
    Save the FAISS index to a file.

    Args:
        index (faiss.Index): FAISS index to save.
        output_path (str): Path to save the index file.
    """
    print(f"Saving the index to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    faiss.write_index(index, output_path)
    print("Index saved successfully.")

if __name__ == "__main__":
    # Paths
    embeddings_path = "output/train_image_embeddings.pt"
    output_index_path = "output/hnsw_index.faiss"

    # Load embeddings
    embeddings, filenames = load_embeddings(embeddings_path)

    # Create and train HNSW index
    hnsw_index = create_hnsw_index(embeddings, ef_construction=200, M=32)

    # Save the index
    save_index(hnsw_index, output_index_path)
    
    print("HNSW index creation pipeline completed successfully.")
