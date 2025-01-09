import os
import torch
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from transformers import ViTModel, ViTFeatureExtractor

class ImageEmbedder:
    def __init__(self, model_name='google/vit-base-patch16-224', device=None):
        """
        Initialize the ImageEmbedder with a ViT model and feature extractor.

        Args:
            model_name (str): Name of the pretrained ViT model to use.
            device (str): Device to run the model ('cuda', 'mps', or 'cpu'). Defaults to GPU (CUDA/MPS) if available.
        """
        self.device = device if device else ('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name).to(self.device)

    def preprocess_image(self, image_path):
        """
        Preprocess a single image for input to the ViT model.

        Args:
            image_path (str): Path to the image file.

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        image = default_loader(image_path)  # Default PIL loader
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        return inputs['pixel_values'].to(self.device)

    def embed_image(self, image_tensor):
        """
        Generate an embedding for a single image tensor.

        Args:
            image_tensor (torch.Tensor): Preprocessed image tensor.

        Returns:
            torch.Tensor: Embedding vector for the image.
        """
        with torch.no_grad():
            outputs = self.model(image_tensor)
        return outputs.last_hidden_state.mean(dim=1)  # Mean pooling over patch embeddings

    def embed_images_from_directory(self, directory):
        """
        Embed all images in a directory.

        Args:
            directory (str): Path to the directory containing images.

        Returns:
            dict: A dictionary where keys are image filenames and values are embeddings.
        """
        embeddings = {}
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
                    image_path = os.path.join(root, file)
                    try:
                        image_tensor = self.preprocess_image(image_path)
                        embeddings[file] = self.embed_image(image_tensor).cpu()
                    except Exception as e:
                        print(f"Failed to process {file}: {e}")
        return embeddings

if __name__ == "__main__":
    # Example usage
    train_directory = "dataset/test/"
    output_file = "output/test_image_embeddings.pt"

    print("Initializing ImageEmbedder...")
    embedder = ImageEmbedder()

    print(f"Embedding images from training directory: {train_directory}")
    embeddings = embedder.embed_images_from_directory(train_directory)

    print(f"Saving embeddings to {output_file}")
    torch.save(embeddings, output_file)
    print("Process completed successfully!")
