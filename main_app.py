
import streamlit as st
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel
import torch
from pymongo import MongoClient, server_api
from urllib.parse import quote_plus
import base64
import io
import math

# Load the Vision Transformer model and feature extractor.
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTModel.from_pretrained('google/vit-base-patch16-224')

# Securely format the MongoDB connection string with username and password.
username = "Ananth"
password = "iamPAWAN@11" 

uri = "mongodb+srv://{username}:{password}@visiontransformer-based.7aqyzdx.mongodb.net/?retryWrites=true&w=majority&appName=VisionTransformer-BasedImageRetrievalSystem".format(
    username=quote_plus(username),
    password=quote_plus(password)
)
client = MongoClient(uri, server_api=server_api.ServerApi('1'))
db = client['Vision_transformer']
features_collection = db['image_retrieval']


# Extract feature vectors from an image using the Vision Transformer.
def extract_features(image):
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    features = outputs.last_hidden_state[:, 0, :].detach().numpy().flatten()
    return features


# Streamlit UI setup
st.title("Image Retrieval System")
st.sidebar.title("Project Information")
st.sidebar.write("""
Welcome to the Image Retrieval System!

This project uses a Vision Transformer (ViT) to extract feature vectors from images, allowing you to find the most similar images based on these features.

### How It Works:
1. **Upload an Image**: Use the file uploader to choose an image for which you want to find similar ones.
2. **Feature Extraction**: The uploaded image is processed through a Vision Transformer to extract its feature vector.
3. **Find Similar Images**: The extracted feature vector is compared against a database to find the top similar images.
4. **Download Results**: You can download the similar images for further use or analysis.

### How to Use This App:
- Use the slider to choose how many similar images you want to see.
- Upload an image to start the search.
- View and download the similar images displayed in the results section.
""")

# Slider for choosing the number of similar images to retrieve.
num_similar_images = st.slider("How many similar images would you like to see?", min_value=1, max_value=10, value=5)

# Radio buttons for selecting the type of search.
search_type = st.radio(
    "Choose the search type:",
    ("Cosine", "Dot Product", "Euclidean Distance")
)

# File uploader for image retrieval.
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], help="Select an image to find similar ones.")


# Process the uploaded image and display it.
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    except Exception as e:
        st.error(f"Error loading image: {e}")

    # Perform the search and handle results.
    with st.spinner("Finding similar images..."):
        try:
            query_features = extract_features(image).tolist()
            # Generate a MongoDB query based on the selected search type.
            query = {}
            if search_type == "Cosine":
                query = {
                    "$vectorSearch": {
                        "index": "image_similarity_search",
                        "path": "feature_vector",
                        "queryVector": query_features,
                        "numCandidates": 10,
                        "limit": num_similar_images
                    }
                }
            elif search_type == "Dot Product":
                query = {
                    "$vectorSearch": {
                        "index": "dotProduct_image_retrieval",
                        "path": "feature_vector",
                        "queryVector": query_features,
                        "numCandidates": 10,
                        "limit": num_similar_images
                    }
                }
            elif search_type == "Euclidean Distance":
                query = {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "feature_vector",
                        "queryVector": query_features,
                        "numCandidates": 10,
                        "limit": num_similar_images
                    }
                }

            results = list(features_collection.aggregate([query]))
            similar_images = [result["image_name"] for result in results]

            st.header(f"Top {num_similar_images} Similar Images")

            num_columns = math.ceil(len(similar_images) / 5)  

             # Display and offer download links for each similar image.
            for i in range(0, len(similar_images), num_columns):
                cols = st.columns(num_columns)
                for j, img_path in enumerate(similar_images[i:i + num_columns]):
                    with cols[j]:
                        st.image(img_path, caption=img_path.split("/")[-1], use_column_width=True)

                        with open(img_path, "rb") as img_file:
                            img_bytes = img_file.read()
                        b64 = base64.b64encode(img_bytes).decode()
                        download_name = img_path.split("/")[-1]

                        st.download_button(
                            label=f"Download {download_name}",
                            data=f"data:image/jpeg;base64,{b64}",
                            file_name=download_name,
                            mime="image/jpeg"
                        )

        except Exception as e:
            st.error(f"Error querying MongoDB: {e}")