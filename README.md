# Vision Transformer-based Advanced Image Retrieval System

## Overview

This project leverages the power of Vision Transformers (ViT) to build an advanced, scalable, and versatile image search system. The system encodes images into a latent space representation, stores these representations in MongoDB, and provides efficient image retrieval using multiple vector search indices. The project also incorporates cutting-edge indexing techniques, such as Product Quantization (PQ) and Hierarchical Navigable Small World (HNSW) graph, to optimize retrieval speed and memory efficiency.

---

## Features

1. **Vision Transformer (ViT) Embedding:**
   - Utilizes the ViT model to encode images into compact 768-dimensional vectors.
   - Embeddings are stored efficiently for scalable retrieval.

2. **Multiple Search Indices:**
   - **Cosine Similarity**
   - **Dot Product**
   - **Euclidean Distance**
   - **Product Quantization (PQ)** for memory-efficient retrieval.
   - **HNSW Graph** for high-speed, approximate nearest neighbor searches.

3. **Streamlit Web Application:**
   - User-friendly interface for uploading images and retrieving visually similar images.
   - Option to choose a similarity metric dynamically.

4. **Dataset:**
   - 4,000 images categorized into four classes: buffalo, elephant, rhino, and zebra.
   - Images sourced from Google, Pexels, Pixabay, and Unsplash.

5. **Comparison of PQ and HNSW Indexing:**
   - Detailed evaluation of index performance in terms of precision, recall, query latency, and memory usage.

---

## Dataset

- **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/ayushv322/animal-classification)
- **Structure:**
  - 4,000 images, divided equally into four folders representing four animal classes.
  - 3,000 images used for training; 1,000 reserved for testing.

---

## Vision Transformer (ViT)

### How ViT Works:
- Images are split into fixed-size patches.
- Patches are embedded and fed into a transformer model with a self-attention mechanism.
- A `CLS` token is used to encode the entire image into a compact vector.

### Why ViT?
- **Attention Mechanism:** Dynamically focuses on relevant image regions.
- **Performance:** Outperforms traditional CNNs on large-scale datasets.
- **Compact Representation:** Produces efficient 768-dimensional vectors.

---

## MongoDB Vector Search

MongoDB provides efficient vector search capabilities, allowing the storage of high-dimensional vectors and retrieval using similarity measures like cosine similarity, Euclidean distance, and dot product. The BSON-based storage ensures scalability and seamless querying.

---

## Indexing Techniques and Comparison

The project implements two advanced FAISS-based indexing techniques: **Product Quantization (PQ)** and **HNSW Graph**.

### Product Quantization (PQ):
- Reduces memory usage by quantizing high-dimensional vectors into subspaces.
- Suitable for scenarios with constrained memory resources.

### HNSW Graph:
- Creates a navigable graph structure for approximate nearest neighbor searches.
- Optimized for high-speed retrieval, especially in large datasets.

### Evaluation Results:

| Method                                | Precision@5 | Recall@5 | Avg Query Latency | Index Size |
|---------------------------------------|-------------|----------|-------------------|------------|
| Product Quantization (PQ)            | 0.4685      | 0.4685   | 0.0000 s          | 0.99 MB    |
| Hierarchical Navigable Small World (HNSW) | 0.9899      | 0.9899   | 0.0000 s          | 12.47 MB   |

#### Key Observations:
- **PQ Index:**
  - Significantly lower memory usage (0.99 MB).
  - Moderate precision and recall.
  - Suitable for memory-constrained systems.

- **HNSW Index:**
  - Superior precision and recall (close to 100%).
  - Higher memory usage (12.47 MB).
  - Ideal for scenarios demanding high accuracy and fast retrieval.

---

## System Architecture

1. **Data Preparation:**
   - Images are processed and embedded into vectors using the ViT model.

2. **Index Creation:**
   - PQ and HNSW indices are built using FAISS.

3. **Database Storage:**
   - Vectors are stored in MongoDB for efficient management and querying.

4. **Web Application:**
   - Streamlit-based interface for seamless interaction.

---

## Streamlit Web Application

- **Features:**
  - Upload an image.
  - Choose a similarity metric.
  - Retrieve top-k similar images instantly.

---

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/username/project_name.git](https://github.com/AnanthaPadmanaban-KrishnaKumar/Vector-Based-Image-Retrieval-System.git)
   cd Vector-Based-Image-Retrieval-System
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the web application:
   ```bash
   streamlit run app.py
   ```

---

## Future Work

1. **Integration of FAISS:**
   - Further optimization of PQ and HNSW indices.
   - Extending the system to handle larger datasets.

2. **Performance Metrics:**
   - Comprehensive analysis of precision/recall and latency across diverse datasets.

3. **Advanced Search Options:**
   - Incorporate multimodal search capabilities combining text and image queries.

---

## Credits

- **Model:** Vision Transformer (ViT) by Google.
- **Libraries:** FAISS, PyTorch, Transformers, MongoDB, Streamlit.
- **Dataset:** Kaggle.

---

## License

This project is licensed under the MIT License. See `MIT` for details.

---

## Contact

For questions or collaboration, reach out to Anantha Padmanaban at [anantha11k@gmail.com].
