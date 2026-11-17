"""FastAPI service for the Schedule Risk Predictor.

Run: uvicorn app:app --reload
POST /predict with a project snapshot taken at roughly 30 percent
complete and receive a slip probability, a predicted delay in days, a
risk band, and the strongest known drivers of slip risk.
"""
from fastapi import FastAPI
from pydantic import BaseModel, Field

from predict import predict

app = FastAPI(title="Schedule Risk Predictor", version="1.0.0")


class Snapshot(BaseModel):
    planned_duration_days: float = Field(gt=0)
    budget_musd: float = Field(gt=0)
    num_activities: int = Field(gt=0)
    dependency_density: float
    critical_path_ratio: float
    subcontractor_count: int
    pct_complete_snapshot: float
    spi: float = Field(gt=0)
    cpi: float = Field(gt=0)
    change_orders_so_far: int
    weather_delay_days_so_far: float
    rework_pct: float
    crew_turnover_pct: float
    procurement_delay_days_so_far: float
    avg_float_days: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def post_predict(snapshot: Snapshot):
    return predict(snapshot.model_dump())
