from urllib import request
from urllib.request import Request

from fastapi import FastAPI, Request

from scripts.schemas import Essay
from scripts.scoring import compute_score
from scripts.model_loader import load_models
from scripts.trainer import train_model

app = FastAPI(
    title="Essay Scoring"
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
        essay.question,
        bert_model,
        ling_model
    )


@app.post("/retrain")
def retrain():

    ok = train_model()

    if ok:
        bert_model, ling_model = load_models()

        app.state.bert_model = bert_model
        app.state.ling_model = ling_model

    return {
        "status": ok
    }