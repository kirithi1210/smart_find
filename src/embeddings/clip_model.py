import os
import io
import base64
import requests
from PIL import Image
from typing import List, Union

class CLIPEmbedder:
    def __init__(self):
        self.api_key = os.getenv("JINA_API_KEY")
        self.url = "https://api.jina.ai/v1/embeddings"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def _get_embedding(self, input_data: dict) -> List[float]:
        payload = {
            "model": "jina-clip-v1",
            "normalized": True,
            "embedding_type": "float",
            "input": [input_data]
        }
        response = requests.post(self.url, headers=self.headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"Vector API Error: {response.text}")
            
        return response.json()["data"][0]["embedding"]

    def get_text_embedding(self, text: str) -> List[float]:
        return self._get_embedding({"text": text})

    def get_image_embedding(self, image_input: Union[str, Image.Image]) -> List[float]:
        if isinstance(image_input, str):
            image = Image.open(image_input)
        else:
            image = image_input

        # Convert image to base64 string for API transfer
        image = image.convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        base64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        return self._get_embedding({"image": f"data:image/jpeg;base64,{base64_str}"})

    def get_multimodal_embedding(self, text: str, image_input: Union[str, Image.Image]) -> List[float]:
        text_vec = self.get_text_embedding(text)
        image_vec = self.get_image_embedding(image_input)
        
        # Average the features to create a single multimodal vector
        combined = [(t + i) / 2.0 for t, i in zip(text_vec, image_vec)]
        
        # Re-normalize the combined vector
        magnitude = sum(x**2 for x in combined) ** 0.5
        return [x / magnitude for x in combined]

# Singleton instance
clip_embedder = CLIPEmbedder()