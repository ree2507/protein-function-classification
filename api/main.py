import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .predict import predict, CLASS_NAMES

app = FastAPI(title="Protein Classifier API", version="1.0.0")

class PredictRequest(BaseModel):
    sequence: str = Field(..., description="Protein amino acid sequence")
    model: str = Field(default="lstm", description="Model to use: cnn, lstm, esm2, or all")

class PredictResponse(BaseModel):
    success: bool
    sequence_length: int
    model_used: str
    prediction: dict = None
    predictions: dict = None
    class_info: dict = None
    error: str = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/models")
def models():
    return {
        "models": ["cnn", "lstm", "esm2"],
        "classes": CLASS_NAMES,
        "default": "lstm"
    }

@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    model_type = req.model.lower()
    if model_type not in ("cnn", "lstm", "esm2", "all"):
        raise HTTPException(status_code=400, detail="Invalid model. Choose: cnn, lstm, esm2, or all")
    result = predict(req.sequence, model_type)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result
