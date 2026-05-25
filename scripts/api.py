from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from scripts.schemas import Essay
from scripts.scoring import compute_score
from scripts.model_loader import load_models

app = FastAPI(
    title="Essay Scoring"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    bert_model, ling_model = load_models()
    app.state.bert_model = bert_model
    app.state.ling_model = ling_model


@app.post("/score")
def score_essay(essay: Essay, request: Request):
    bert_model = request.app.state.bert_model
    ling_model = request.app.state.ling_model

    return compute_score(
        essay.answer,
        essay.reference,
        bert_model,
        ling_model
    )
