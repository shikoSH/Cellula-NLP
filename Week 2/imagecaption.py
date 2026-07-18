from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

_processor = None
_model = None

def load_model():
    global _processor, _model
    if _model is None:
        _processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return _processor, _model

def generate_caption(image: Image.Image) -> str:
    processor, model = load_model()
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=30)
    return processor.decode(out[0], skip_special_tokens=True)