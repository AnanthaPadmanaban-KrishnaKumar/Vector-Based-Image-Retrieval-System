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

def create_pq_index(vectors, nlist=100, m=8):
    """
    Create a Product Quantization (PQ) index for the given embeddings.

    Args:
        vectors (np.ndarray): Embedding vectors.
        nlist (int): Number of clusters.
        m (int): Number of subquantizers.

    Returns:
        faiss.IndexIVFPQ: Trained PQ index.
    """
    d = vectors.shape[1]  # Dimension of the embeddings
    print(f"Creating a PQ index with {nlist} clusters and {m} subquantizers...")

    # Ensure embedding dimension is divisible by the number of subquantizers
    if d % m != 0:
        raise ValueError(f"Embedding dimension {d} must be divisible by the number of subquantizers {m}.")

    # Define quantizer and PQ index
    quantizer = faiss.IndexFlatL2(d)  # L2 distance metric
    pq_index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)  # PQ index with 8 bits per subvector

    # Train the PQ index
    print("Training the PQ index...")
    pq_index.train(vectors)
    print("PQ index training completed.")

    # Add embeddings to the PQ index
    print("Adding embeddings to the PQ index...")
    pq_index.add(vectors)
    print(f"Added {pq_index.ntotal} embeddings to the index.")

    return pq_index

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
    output_index_path = "output/pq_index.faiss"

    # Load embeddings
    embeddings, filenames = load_embeddings(embeddings_path)

    # Create and train PQ index
    pq_index = create_pq_index(embeddings, nlist=50, m=16)

    # Save the index
    save_index(pq_index, output_index_path)

    print("Product Quantization index creation pipeline completed successfully.")
