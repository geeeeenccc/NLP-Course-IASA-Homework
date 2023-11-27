import re
import emoji
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForTokenClassification
from optimum.pipelines import pipeline

tokenizer_name = 'bert-base-uncased'

label2id = {'O': 0, 'B-LOC': 1, 'I-LOC': 2}
id2label = {v: k for k, v in label2id.items()}

class Prediction:
    """
    Wrapper for Location Prediction NER model.
    """

    def __init__(self, chkp_path: str, device: str, thresh: float):
        self.thresh = thresh

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Note: Load the ONNX model using ORTModelForTokenClassification
        self.base_model = ORTModelForTokenClassification.from_pretrained(chkp_path, file='model.onnx')

        # Note: Create the model using pipeline from optimum.pipelines
        self.model = pipeline(
            'token-classification',
            model=self.base_model,
            tokenizer=self.tokenizer,
            aggregation_strategy='simple',
            device=device
        )

    @staticmethod
    def filter_text(text: str) -> str:
        text = emoji.replace_emoji(text)
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r' +', ' ', text)
        return text

    @staticmethod
    def post_process(locations: List[str]) -> List[str]:
        # Note: 'l == 'RU'' to match the format in the list
        locations = [l for l in locations if (len(l) > 3 and any(c.isupper() for c in l)) or l == 'RU' or l == 'USA' or l == 'RB']
        return locations

    def predict(self, texts: List[str]) -> List[List[str]]:
        filtered_texts = [
            LocationPredictor.filter_text(text)
            for text in texts
        ]

        all_locations = self.model(filtered_texts)

        filtered_locations = [
            [pred['word'] for pred in preds if pred['score'] >= self.thresh]
            for preds in all_locations
        ]

        return [LocationPredictor.post_process(locations)
                for locations in filtered_locations]


app = FastAPI()

Prediction_model = Prediction('models/model/', 'cpu', 0.5)  # Change 'cpu' to 'cuda'

class Input(BaseModel):
    texts: List[str]  # Use List[str] instead of list[str]


@app.post('/locations')
async def get_locations(input: Input):
    return Prediction_model.predict(input.texts)
