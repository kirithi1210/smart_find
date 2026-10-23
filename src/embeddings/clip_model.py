from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np
from typing import Union, List

class CLIPEmbeddingModel:
    def __init__(self, model_name: str = "clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)

    def get_text_embedding(self, text: str) -> List[float]:
        embedding = self.model.encode(text, convert_to_numpy=True)
        norm_embedding = embedding / np.linalg.norm(embedding)
        return norm_embedding.tolist()

    def get_image_embedding(self, image_input: Union[str, Image.Image]) -> List[float]:
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")

        embedding = self.model.encode(image, convert_to_numpy=True)
        norm_embedding = embedding / np.linalg.norm(embedding)
        return norm_embedding.tolist()

    def get_multimodal_embedding(self, text: str, image_input: Union[str, Image.Image]) -> List[float]:
        text_emb = np.array(self.get_text_embedding(text))
        img_emb = np.array(self.get_image_embedding(image_input))
        
        combined_emb = (text_emb + img_emb) / 2.0
        norm_combined = combined_emb / np.linalg.norm(combined_emb)
        return norm_combined.tolist()

clip_embedder = CLIPEmbeddingModel() 