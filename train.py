"""Train and evaluate the Schedule Risk Predictor models.

Two tasks share one feature set (signals known at roughly 30 percent
complete, see generate_data.py):
  classification  will_slip: will the project finish more than 14 days late
  regression      delay_days: how many days late it will finish

Three model families are compared on a held out test set with a fixed
split, and the winner of each task is saved to model/ with its metrics.
Evaluation is honest by construction: the test set is never touched
during training, and every reported number comes from that test set.
"""
import json
import os

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, precision_score, recall_score, r2_score, roc_auc_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "planned_duration_days", "budget_musd", "num_activities",
    "dependency_density", "critical_path_ratio", "subcontractor_count",
    "pct_complete_snapshot", "spi", "cpi", "change_orders_so_far",
    "weather_delay_days_so_far", "rework_pct", "crew_turnover_pct",
    "procurement_delay_days_so_far", "avg_float_days",
]


def main():
    df = pd.read_csv("data/projects.csv")
    X, y_cls, y_reg = df[FEATURES], df["will_slip"], df["delay_days"]
    X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
        X, y_cls, y_reg, test_size=0.2, random_state=42, stratify=y_cls
    )

    classifiers = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=42),
    }
    cls_results = {}
    for name, model in classifiers.items():
        model.fit(X_tr, yc_tr)
        prob = model.predict_proba(X_te)[:, 1]
        pred = (prob >= 0.5).astype(int)
        cls_results[name] = {
            "roc_auc": round(roc_auc_score(yc_te, prob), 3),
            "accuracy": round(accuracy_score(yc_te, pred), 3),
            "precision": round(precision_score(yc_te, pred), 3),
            "recall": round(recall_score(yc_te, pred), 3),
            "f1": round(f1_score(yc_te, pred), 3),
            "_model": model,
        }
    best_cls_name = max(cls_results, key=lambda k: cls_results[k]["roc_auc"])
    best_cls = cls_results[best_cls_name].pop("_model")
    for v in cls_results.values():
        v.pop("_model", None)

    regressors = {
        "linear_regression": make_pipeline(StandardScaler(), LinearRegression()),
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=42),
    }
    reg_results = {}
    for name, model in regressors.items():
        model.fit(X_tr, yr_tr)
        pred = model.predict(X_te)
        reg_results[name] = {
            "mae_days": round(mean_absolute_error(yr_te, pred), 2),
            "rmse_days": round(root_mean_squared_error(yr_te, pred), 2),
            "r2": round(r2_score(yr_te, pred), 3),
            "_model": model,
        }
    best_reg_name = min(reg_results, key=lambda k: reg_results[k]["mae_days"])
    best_reg = reg_results[best_reg_name].pop("_model")
    for v in reg_results.values():
        v.pop("_model", None)

    perm = permutation_importance(best_cls, X_te, yc_te, n_repeats=10, random_state=42)
    importances = sorted(
        zip(FEATURES, perm.importances_mean), key=lambda t: t[1], reverse=True
    )
    top_features = [(f, round(float(v), 4)) for f, v in importances[:6]]

    os.makedirs("model", exist_ok=True)
    joblib.dump(best_cls, "model/slip_classifier.joblib")
    joblib.dump(best_reg, "model/delay_regressor.joblib")
    metrics = {
        "dataset_rows": int(len(df)),
        "test_rows": int(len(X_te)),
        "features": FEATURES,
        "classification": {"winner": best_cls_name, "results": cls_results},
        "regression": {"winner": best_reg_name, "results": reg_results},
        "top_features_for_slip": top_features,
        "slip_definition": "delay_days greater than 14",
        "snapshot": "features measured at roughly 30 percent complete",
    }
    with open("model/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps({k: metrics[k] for k in ("classification", "regression", "top_features_for_slip")}, indent=2))


if __name__ == "__main__":
    main()
