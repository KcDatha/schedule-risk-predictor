"""Smoke test: a visibly troubled project must score riskier than a
healthy one, and all outputs must stay in sane ranges."""
from predict import predict

HEALTHY = {
    "planned_duration_days": 720, "budget_musd": 220, "num_activities": 8000,
    "dependency_density": 1.8, "critical_path_ratio": 0.2,
    "subcontractor_count": 18, "pct_complete_snapshot": 30, "spi": 1.02,
    "cpi": 1.01, "change_orders_so_far": 2, "weather_delay_days_so_far": 1,
    "rework_pct": 1.0, "crew_turnover_pct": 6,
    "procurement_delay_days_so_far": 0, "avg_float_days": 28,
}
TROUBLED = {
    "planned_duration_days": 720, "budget_musd": 220, "num_activities": 8000,
    "dependency_density": 2.9, "critical_path_ratio": 0.4,
    "subcontractor_count": 34, "pct_complete_snapshot": 30, "spi": 0.78,
    "cpi": 0.86, "change_orders_so_far": 16, "weather_delay_days_so_far": 22,
    "rework_pct": 8.5, "crew_turnover_pct": 30,
    "procurement_delay_days_so_far": 41, "avg_float_days": 3,
}

h, t = predict(HEALTHY), predict(TROUBLED)
for r in (h, t):
    assert 0.0 <= r["slip_probability"] <= 1.0
    assert -40 <= r["predicted_delay_days"] <= 460
    assert r["risk_band"] in ("low", "medium", "high")
assert t["slip_probability"] > h["slip_probability"], "troubled project must score riskier"
assert t["predicted_delay_days"] > h["predicted_delay_days"]
print("healthy:", h)
print("troubled:", t)
print("SMOKE TEST PASSED")
