import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from scripts.config import MODEL_NAME, BATCH_SIZE, MAX_LENGTH as MAX_LEN, device

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(MODEL_NAME)

bert_model = torch.quantization.quantize_dynamic(
    bert_model, {torch.nn.Linear}, dtype=torch.qint8
)

bert_model.to(device)
bert_model.eval()

def get_bert_embeddings(questions, answers):
    embeddings = []
    
    for i in range(0, len(questions), BATCH_SIZE):
        q_batch = list(questions[i:i + BATCH_SIZE])
        a_batch = list(answers[i:i + BATCH_SIZE])

        inputs = tokenizer(
            q_batch,
            a_batch,
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
            emb = (token_embeddings * mask).sum(1) / mask.sum(1)
        embeddings.append(emb.cpu().numpy())
    return np.vstack(embeddings)