from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch
import sys
import os

# Add model directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../model'))
from hgat import HGAT
from dataset import FakeNewsDataset

app = FastAPI(title="HGAT Fake News Detection API")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model for inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HGAT().to(device)
model.eval()

# We use the dataset's tokenizer logic
dataset_helper = FakeNewsDataset()

class ArticleRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    beta: float

@app.post("/predict", response_model=PredictionResponse)
def predict(request: ArticleRequest):
    try:
        # Preprocess Text
        encoding = dataset_helper.tokenizer(
            request.text,
            add_special_tokens=True,
            max_length=dataset_helper.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Construct Graph (placeholder logic for entities)
        num_nodes = 5
        x_graph = torch.randn(num_nodes, 768)
        edge_index = torch.tensor([
            [0, 1, 1, 2, 3, 4],
            [1, 0, 2, 1, 4, 3]
        ], dtype=torch.long)
        batch = torch.zeros(num_nodes, dtype=torch.long) # Single graph in batch
        
        # Move to device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        x_graph = x_graph.to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device)
        
        # Inference
        with torch.no_grad():
            logits, beta = model(input_ids, attention_mask, x_graph, edge_index, batch)
            
            # Softmax for probabilities
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
            
        # Map index to class: 0 -> REAL, 1 -> FAKE
        predicted_class = "REAL" if pred_idx.item() == 0 else "FAKE"
        
        return PredictionResponse(
            prediction=predicted_class,
            confidence=round(conf.item() * 100, 2),
            beta=round(beta.item(), 3)
        )
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
