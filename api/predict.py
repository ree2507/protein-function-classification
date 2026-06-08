import re
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from .model_loader import (
    device, ProteinDataset, ProteinMultiScaleCNN, ProteinLSTM,
    aa_to_int, VOCAB_SIZE, MAX_LEN, load_cnn, load_lstm, load_esm2
)

with open('data/processed/label_mapping.json', 'r') as f:
    LABEL_MAPPING = json.load(f)
CLASS_NAMES = [LABEL_MAPPING[str(i)] for i in range(len(LABEL_MAPPING))]

with open('data/class_info.json', 'r') as f:
    CLASS_INFO = json.load(f)

AMINO_ACIDS_SET = set('ACDEFGHIKLMNPQRSTVWY')
SEQ_PATTERN = re.compile(r'^[ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy]{4,1000}$')


def validate_sequence(seq):
    if not SEQ_PATTERN.match(seq):
        return False
    return True


def preprocess(seq):
    return seq.upper()


def predict_cnn(seq):
    model = load_cnn()
    seq_int = [[aa_to_int.get(aa, 0) for aa in seq[:MAX_LEN]]]
    if len(seq_int[0]) < MAX_LEN:
        seq_int[0] += [0] * (MAX_LEN - len(seq_int[0]))
    x = torch.tensor(seq_int, dtype=torch.long).to(device)
    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
    return probs


def predict_lstm(seq):
    model = load_lstm()
    seq_int = [[aa_to_int.get(aa, 0) for aa in seq[:MAX_LEN]]]
    if len(seq_int[0]) < MAX_LEN:
        seq_int[0] += [0] * (MAX_LEN - len(seq_int[0]))
    x = torch.tensor(seq_int, dtype=torch.long).to(device)
    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
    return probs


def predict_esm2(seq):
    model, tokenizer = load_esm2()
    encoded = tokenizer(seq, padding="max_length", truncation=True,
                        max_length=1002, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.amp.autocast('cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    return probs


def format_prediction(probs, model_name):
    pred_label = int(np.argmax(probs))
    confidence = float(probs[pred_label])
    return {
        "class": CLASS_NAMES[pred_label],
        "label": pred_label,
        "confidence": round(confidence, 4),
        "probabilities": {
            CLASS_NAMES[i]: round(float(probs[i]), 4)
            for i in range(len(CLASS_NAMES))
        }
    }


def get_class_info(label):
    key = str(label)
    info = CLASS_INFO.get(key, {})
    return info


def predict(seq, model_type="lstm"):
    seq = preprocess(seq)
    if not validate_sequence(seq):
        return {"success": False, "error": "Invalid sequence. Must be 4-1000 amino acid characters (A-Z)."}

    if model_type == "all":
        results = {}
        for name, func in [("cnn", predict_cnn), ("lstm", predict_lstm), ("esm2", predict_esm2)]:
            probs = func(seq)
            results[name] = format_prediction(probs, name)
        pred_label = results["lstm"]["label"]
        return {
            "success": True,
            "sequence_length": len(seq),
            "model_used": "all",
            "predictions": results,
            "class_info": get_class_info(pred_label)
        }

    predictors = {
        "cnn": predict_cnn,
        "lstm": predict_lstm,
        "esm2": predict_esm2
    }
    func = predictors.get(model_type, predict_lstm)
    probs = func(seq)
    pred = format_prediction(probs, model_type)
    return {
        "success": True,
        "sequence_length": len(seq),
        "model_used": model_type,
        "prediction": pred,
        "class_info": get_class_info(pred["label"])
    }
