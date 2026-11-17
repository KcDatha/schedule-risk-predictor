"""Generate the Schedule Risk Predictor dataset.

The dataset is synthetic and generated from a documented process built on
earned value management relationships: projects that are already behind
schedule (low SPI) or over budget (low CPI) at an early snapshot, with many
change orders, procurement delays, rework, and little schedule float, tend
to finish late. Every feature is measured at a snapshot taken when the
project is roughly 30 percent complete, so a model trained on this data
only ever sees information that would genuinely be available early in a
real project. No feature is derived from the final outcome (no leakage).

Output: data/projects.csv with two targets:
  delay_days  final delay beyond the planned duration, in days (regression)
  will_slip   1 when delay_days is greater than 14, else 0 (classification)
"""
import numpy as np
import pandas as pd

SEED = 42
N = 2400


def main():
    rng = np.random.default_rng(SEED)

    planned_duration_days = rng.integers(180, 1460, N).astype(float)
    budget_musd = np.round(rng.uniform(5, 850, N), 1)
    num_activities = rng.integers(200, 100000, N)
    dependency_density = np.round(rng.uniform(1.1, 3.4, N), 2)
    critical_path_ratio = np.round(rng.uniform(0.08, 0.45, N), 3)
    subcontractor_count = rng.integers(4, 60, N)
    pct_complete_snapshot = np.round(rng.uniform(24, 36, N), 1)

    # Early performance signals. Centered near 1.0 (on plan), realistically spread.
    spi = np.clip(rng.normal(0.97, 0.09, N), 0.62, 1.18)
    cpi = np.clip(rng.normal(0.98, 0.08, N), 0.66, 1.16)
    change_orders = rng.poisson(6, N)
    weather_delay_days = np.round(np.clip(rng.normal(6, 7, N), 0, 45), 1)
    rework_pct = np.round(np.clip(rng.normal(3.2, 2.4, N), 0, 15), 2)
    crew_turnover_pct = np.round(np.clip(rng.normal(12, 8, N), 0, 55), 1)
    procurement_delay_days = np.round(np.clip(rng.normal(9, 11, N), 0, 70), 1)
    avg_float_days = np.round(np.clip(rng.normal(16, 10, N), 0, 60), 1)

    # Latent schedule risk from the early signals (standardized effects).
    risk = (
        2.6 * (1.0 - spi)
        + 1.4 * (1.0 - cpi)
        + 0.045 * change_orders
        + 0.016 * weather_delay_days
        + 0.085 * rework_pct
        + 0.28 * (dependency_density - 2.0)
        + 0.9 * (critical_path_ratio - 0.25)
        + 0.014 * procurement_delay_days
        + 0.010 * crew_turnover_pct
        - 0.012 * avg_float_days
        - 0.85
        + rng.normal(0, 0.16, N)
    )

    # Delay grows with risk and with project length, then noise.
    delay_days = planned_duration_days * 0.13 * risk + rng.normal(0, 9, N)
    delay_days = np.round(np.clip(delay_days, -25, 420), 0)

    df = pd.DataFrame(
        {
            "planned_duration_days": planned_duration_days,
            "budget_musd": budget_musd,
            "num_activities": num_activities,
            "dependency_density": dependency_density,
            "critical_path_ratio": critical_path_ratio,
            "subcontractor_count": subcontractor_count,
            "pct_complete_snapshot": pct_complete_snapshot,
            "spi": np.round(spi, 3),
            "cpi": np.round(cpi, 3),
            "change_orders_so_far": change_orders,
            "weather_delay_days_so_far": weather_delay_days,
            "rework_pct": rework_pct,
            "crew_turnover_pct": crew_turnover_pct,
            "procurement_delay_days_so_far": procurement_delay_days,
            "avg_float_days": avg_float_days,
            "delay_days": delay_days,
        }
    )
    df["will_slip"] = (df["delay_days"] > 14).astype(int)

    df.to_csv("data/projects.csv", index=False)
    print("rows:", len(df))
    print("slip rate:", round(df["will_slip"].mean(), 3))
    print("delay days mean:", round(df["delay_days"].mean(), 1),
          "| median:", df["delay_days"].median(),
          "| max:", df["delay_days"].max())


if __name__ == "__main__":
    main()
