"""Run the whole project: generate data -> demand forecast -> expiry risk -> summary."""
import json
from pathlib import Path

import pandas as pd

import data_generation
import demand_forecasting
import expiry_risk

ROOT = Path(__file__).parent
DATA, OUT, MODELS = ROOT / "data", ROOT / "outputs", ROOT / "models"


def main():
    OUT.mkdir(exist_ok=True)
    print("1/3  Generating synthetic pharmacy data ...")
    meds, sales = data_generation.generate(DATA)
    print(f"     {len(meds)} medicines, {len(sales):,} daily rows "
          f"({sales['date'].min().date()} to {sales['date'].max().date()})")

    print("2/3  Training demand forecaster ...")
    dres, forecast = demand_forecasting.run(DATA, OUT, MODELS)
    o = dres["overall"]
    print(f"     WAPE  model {o['model']['WAPE_%']:.1f}%  |  same-weekday baseline "
          f"{o['baseline_same_weekday']['WAPE_%']:.1f}%  |  28-day average "
          f"{o['baseline_moving_avg_28']['WAPE_%']:.1f}%")

    print("3/3  Training expiry-risk model ...")
    eres, report = expiry_risk.run(DATA, OUT, MODELS, forecast)
    cs = eres["current_stock"]
    print(f"     ROC-AUC {eres['ROC_AUC']:.3f}  PR-AUC {eres['PR_AUC']:.3f}  "
          f"(days-of-cover rule: {eres['baseline_coverage_rule_ROC_AUC']:.3f})")
    print(f"     Live batches: {cs['live_batches']}  tiers: {cs['tier_counts']}  "
          f"projected write-off: INR {cs['value_at_risk_inr']:,.0f}")

    print("\nTop exposures:")
    top = report[report.risk_tier != "Low"].head(6)
    print(top[["medicine", "batch_id", "expiry_date", "units_remaining", "expiry_risk_prob",
               "expected_units_expiring", "value_at_risk_inr", "risk_tier"]].to_string(index=False))
    print(f"\nOutputs written to {OUT}")


if __name__ == "__main__":
    main()
