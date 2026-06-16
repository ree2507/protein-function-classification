import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .predict import predict, CLASS_NAMES

app = FastAPI(title="Protein Classifier API (LSTM)", version="1.0.0")

class PredictRequest(BaseModel):
    sequence: str = Field(..., description="Protein amino acid sequence")
    model: str = Field(default="lstm", description="Model to use (only lstm supported)")

class PredictResponse(BaseModel):
    success: bool
    sequence_length: int
    model_used: str
    prediction: dict = None
    class_info: dict = None
    error: str = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/models")
def models():
    return {
        "models": ["lstm"],
        "classes": CLASS_NAMES,
        "default": "lstm"
    }

@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    model_type = req.model.lower()
    if model_type != "lstm":
        raise HTTPException(status_code=400, detail="Only 'lstm' model is supported.")
    result = predict(req.sequence)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result
