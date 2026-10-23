import torch
import clip
from PIL import Image
from typing import List, Union

class CLIPEmbedder:
    def __init__(self):
        self.model = None
        self.preprocess = None
        self.device = "cpu"

    def _load_model(self):
        if self.model is None:
            # Loads model into memory only when inference is requested
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def get_text_embedding(self, text: str) -> List[float]:
        self._load_model()
        text_inputs = clip.tokenize([text]).to(self.device)
        with torch.no-grad():
            text_features = self.model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()[0].tolist()

    def get_image_embedding(self, image_input: Union[str, Image.Image]) -> List[float]:
        self._load_model()
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")
            
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no-grad():
            image_features = self.model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()[0].tolist()

    def get_multimodal_embedding(self, text: str, image_input: Union[str, Image.Image]) -> List[float]:
        """Combines image and text into a single embedding vector."""
        # Get individual embeddings as tensors
        text_features = torch.tensor(self.get_text_embedding(text))
        image_features = torch.tensor(self.get_image_embedding(image_input))
        
        # Average the features and re-normalize
        combined_features = (text_features + image_features) / 2.0
        combined_features /= combined_features.norm(dim=-1, keepdim=True)
        
        return combined_features.tolist()

# Singleton instance
clip_embedder = CLIPEmbedder()