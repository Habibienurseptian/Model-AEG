import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from scripts.config import MODEL_NAME, BATCH_SIZE, MAX_LENGTH as MAX_LEN, device

tokenizer = None
bert_model = None

def load_model():
    global tokenizer, bert_model
    if tokenizer is None or bert_model is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        model.to(device)
        model.eval()
        bert_model = model
    return tokenizer, bert_model


def get_bert_embeddings(texts):
    global tokenizer, bert_model
    tokenizer, bert_model = load_model()
    embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = bert_model(**inputs)
            token_embeddings = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            denom = mask.sum(1).clamp(min=1e-9)
            emb = (token_embeddings * mask).sum(1) / denom
        embeddings.append(emb.cpu().numpy())

    return np.vstack(embeddings)