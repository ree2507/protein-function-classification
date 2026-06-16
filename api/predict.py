import re
import json
import numpy as np
import torch

from .model_loader import (
    device, aa_to_int, VOCAB_SIZE, MAX_LEN, load_lstm
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


def predict(seq):
    seq = preprocess(seq)
    if not validate_sequence(seq):
        return {"success": False, "error": "Invalid sequence. Must be 4-1000 amino acid characters (A-Z)."}

    probs = predict_lstm(seq)
    pred = format_prediction(probs, "lstm")
    return {
        "success": True,
        "sequence_length": len(seq),
        "model_used": "lstm",
        "prediction": pred,
        "class_info": get_class_info(pred["label"])
    }
