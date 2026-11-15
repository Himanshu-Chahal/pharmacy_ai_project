"""
Expiry-risk model.

Question answered: "Given what is on the shelf today, which batches will NOT be
sold before they expire?"

Two complementary views are combined:

1. Classifier (learned from history)
   Every week each live batch is snapshotted with features such as days left,
   units left, units in earlier-expiring batches (they are dispensed first, FEFO),
   recent sales rate and how many days of stock that represents. The label is
   whether the batch ultimately expired with units unsold.

2. Forecast-based projection (uses the demand model)
   Expected demand until the expiry date, taken from the 28-day forecast, is
   consumed first-expiry-first-out. Whatever stock remains after that is the
   expected number of units to expire.

The two are merged into a risk tier and a suggested action per batch.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score, roc_curve

CUTOFF = pd.Timestamp("2026-03-31")   # train on outcomes known by here, test on snapshots after
FEATURES = ["days_to_expiry", "units_remaining", "units_ahead", "rate7", "rate28",
            "coverage_ratio", "days_of_cover"]
# risk may only go UP as days of stock cover goes up (stops the model learning shortcuts
# such as "few units left => safe", which is wrong when earlier batches queue ahead of it)
MONOTONIC = {"coverage_ratio": 1, "days_of_cover": 1}

TIER_ACTIONS = {
    "High": "Act now: discount, transfer to another store or return to supplier; pause reorders",
    "Medium": "Watch: dispense this batch first and shrink the next order",
    "Low": "No action needed",
}


def add_features(df, meds):
    df = df.merge(meds[["medicine_id", "medicine", "category_id", "unit_cost"]], on="medicine_id")
    df["frac_remaining"] = df["units_remaining"] / df["units_received"]
    df["days_of_cover"] = (df["units_remaining"] + df["units_ahead"]) / (df["rate28"] + 0.1)
    df["coverage_ratio"] = df["days_of_cover"] / df["days_to_expiry"]
    df["snapshot_month"] = pd.to_datetime(df["snapshot_date"]).dt.month
    return df


def load_labelled(data_dir, meds):
    snaps = pd.read_csv(Path(data_dir) / "expiry_snapshots.csv", parse_dates=["snapshot_date"])
    batches = pd.read_csv(Path(data_dir) / "batches.csv", parse_dates=["resolved_date", "expiry_date"])
    df = add_features(snaps, meds)
    df = df.merge(batches[["batch_id", "status", "qty_expired", "resolved_date"]], on="batch_id")
    df = df[df.status != "open"].copy()               # outcome not known yet
    df["label"] = (df["qty_expired"] > 0).astype(int)
    return df


def make_model():
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=250, max_leaf_nodes=15, min_samples_leaf=50,
        l2_regularization=1.0,
        monotonic_cst=[MONOTONIC.get(f, 0) for f in FEATURES], random_state=0)


def evaluate(train, test, proba):
    y = test["label"].to_numpy()
    base = test["coverage_ratio"].to_numpy()
    out = {"train_snapshots": int(len(train)), "test_snapshots": int(len(test)),
           "train_positive_rate": float(train.label.mean()), "test_positive_rate": float(y.mean()),
           "test_expired_batches": int(test.loc[test.label == 1, "batch_id"].nunique()),
           "ROC_AUC": float(roc_auc_score(y, proba)),
           "PR_AUC": float(average_precision_score(y, proba)),
           "baseline_coverage_rule_ROC_AUC": float(roc_auc_score(y, base)),
           "baseline_coverage_rule_PR_AUC": float(average_precision_score(y, base))}
    for thr in (0.3, 0.5):
        pred = (proba >= thr).astype(int)
        out[f"precision@{thr}"] = float(precision_score(y, pred, zero_division=0))
        out[f"recall@{thr}"] = float(recall_score(y, pred, zero_division=0))
    out["ROC_AUC_by_days_to_expiry"] = {}
    for label, lo, hi in [("<=30 days", 0, 30), ("31-90 days", 31, 90), (">90 days", 91, 10_000)]:
        m = (test.days_to_expiry >= lo) & (test.days_to_expiry <= hi)
        if m.sum() > 50 and test.loc[m, "label"].nunique() == 2:
            out["ROC_AUC_by_days_to_expiry"][label] = float(roc_auc_score(test.loc[m, "label"], proba[m.to_numpy()]))
    return out


def project_expiry(current, forecast):
    """Consume forecast demand FEFO and return units projected to expire per batch."""
    f = forecast.pivot(index="horizon", columns="medicine_id", values="forecast_units")
    current = current.sort_values(["medicine_id", "days_to_expiry", "batch_id"]).copy()
    demand_until = []
    for _, r in current.iterrows():
        daily = f[r.medicine_id].to_numpy()
        d = int(r.days_to_expiry)
        horizon_part = daily[: min(d, len(daily))].sum()
        tail = max(0, d - len(daily)) * daily.mean()       # beyond 28 days: assume average level
        demand_until.append(horizon_part + tail)
    current["forecast_demand_until_expiry"] = np.round(demand_until, 1)
    stock_through_batch = current["units_remaining"] + current["units_ahead"]
    current["expected_units_expiring"] = np.clip(
        stock_through_batch - current["forecast_demand_until_expiry"], 0, current["units_remaining"]).round(1)
    return current


def tier(row):
    share = row.expected_units_expiring / row.units_remaining
    if row.expiry_risk_prob >= 0.6 or share >= 0.5:
        return "High"
    if row.expiry_risk_prob >= 0.3 or share >= 0.2:
        return "Medium"
    return "Low"


def plot_overview(test, proba, report, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
    y = test["label"].to_numpy()
    fpr, tpr, _ = roc_curve(y, proba)
    fb, tb, _ = roc_curve(y, test["coverage_ratio"].to_numpy())
    a1.plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"Model (AUC {roc_auc_score(y, proba):.2f})")
    a1.plot(fb, tb, color="#f59e0b", lw=2, ls="--",
            label=f"Days-of-cover rule (AUC {roc_auc_score(y, test['coverage_ratio']):.2f})")
    a1.plot([0, 1], [0, 1], color="#d1d5db", lw=1)
    a1.set_xlabel("False positive rate")
    a1.set_ylabel("True positive rate")
    a1.set_title("Spotting batches that will expire (held-out)", loc="left", fontweight="bold", fontsize=11)
    a1.legend(frameon=False, loc="lower right")

    top = report[(report.risk_tier != "Low") & (report.value_at_risk_inr > 0)].nlargest(10, "value_at_risk_inr").iloc[::-1]
    colors = top.risk_tier.map({"High": "#dc2626", "Medium": "#f59e0b"})
    labels = top.medicine + " (b" + top.batch_id.astype(str) + ")"
    a2.barh(labels, top.value_at_risk_inr, color=colors)
    a2.set_xlabel("Projected write-off value (INR)")
    a2.set_title("Largest expiry exposures in current stock", loc="left", fontweight="bold", fontsize=11)
    for ax in (a1, a2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run(data_dir, out_dir, model_dir, forecast):
    out_dir, model_dir = Path(out_dir), Path(model_dir)
    meds = pd.read_csv(Path(data_dir) / "medicines.csv")
    meds["category_id"] = meds["category"].astype("category").cat.codes

    labelled = load_labelled(data_dir, meds)
    train = labelled[labelled.resolved_date <= CUTOFF]
    test = labelled[labelled.snapshot_date > CUTOFF]

    model = make_model().fit(train[FEATURES], train["label"])
    proba = model.predict_proba(test[FEATURES])[:, 1]
    results = evaluate(train, test, proba)

    # refit on every resolved batch, then score what is on the shelf today
    final = make_model().fit(labelled[FEATURES], labelled["label"])
    current = pd.read_csv(Path(data_dir) / "current_stock.csv", parse_dates=["snapshot_date"])
    current = add_features(current, meds)
    current["expiry_risk_prob"] = final.predict_proba(current[FEATURES])[:, 1].round(3)
    current = project_expiry(current, forecast)
    current["expiry_date"] = (current["snapshot_date"] + pd.to_timedelta(current["days_to_expiry"], unit="D")).dt.date
    current["value_at_risk_inr"] = (current["expected_units_expiring"] * current["unit_cost"]).round(0)
    current["risk_tier"] = current.apply(tier, axis=1)
    current["recommended_action"] = current["risk_tier"].map(TIER_ACTIONS)

    cols = ["medicine", "batch_id", "expiry_date", "days_to_expiry", "units_remaining", "units_ahead",
            "rate28", "expiry_risk_prob", "forecast_demand_until_expiry", "expected_units_expiring",
            "value_at_risk_inr", "risk_tier", "recommended_action"]
    report = current[cols].copy()
    report["rate28"] = report["rate28"].round(1)
    report = report.sort_values(["risk_tier", "value_at_risk_inr"],
                                key=lambda s: s.map({"High": 0, "Medium": 1, "Low": 2}) if s.name == "risk_tier" else -s)
    report.to_csv(out_dir / "expiry_risk_report.csv", index=False)

    results["current_stock"] = {
        "live_batches": int(len(report)),
        "tier_counts": report.risk_tier.value_counts().to_dict(),
        "value_at_risk_inr": float(report.value_at_risk_inr.sum()),
        "high_tier_value_at_risk_inr": float(report.loc[report.risk_tier == "High", "value_at_risk_inr"].sum()),
    }
    plot_overview(test, proba, report, out_dir / "expiry_risk_overview.png")

    import joblib
    joblib.dump(final, model_dir / "expiry_model.joblib")
    (out_dir / "expiry_metrics.json").write_text(json.dumps(results, indent=2))
    return results, report


if __name__ == "__main__":
    root = Path(__file__).parent
    fc = pd.read_csv(root / "outputs" / "forecast_next_28d.csv", parse_dates=["date"])
    res, rep = run(root / "data", root / "outputs", root / "models", fc)
    print(json.dumps(res, indent=2))
