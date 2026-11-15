"""
Demand forecasting: one global gradient-boosting model that predicts daily unit
sales for every medicine, 1-28 days ahead ("direct multi-horizon" forecasting).

Each training row = (medicine, forecast origin date, horizon h):
    features known at the origin  ->  units sold on origin + h

Features
  * recent level / trend      last day, 7/14/28/56-day means, 28-day std
  * same-weekday average      mean of the last 4 same-weekday sales before the origin
  * last-year seasonality     sales around the same date a year earlier (364 days back)
  * calendar of target day    weekday, month, day-of-year (sin/cos)
  * planned promotion flag    promotions are known in advance
  * medicine / category id    categorical

Evaluation is strictly chronological: the model never sees a target date later
than TRAIN_END, and is scored on forecasts issued after TRAIN_END.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

HORIZON = 28
WARMUP = 56
TRAIN_END = pd.Timestamp("2026-07-31")

FEATURES = ["medicine_id", "category_id", "horizon", "target_dow", "target_month",
            "doy_sin", "doy_cos", "promo_target", "last1", "mean7", "mean14", "mean28",
            "mean56", "std28", "same_dow_mean", "trend_ratio", "momentum_ratio",
            "ly_window", "ly_ratio"]
CATEGORICAL = ["medicine_id", "category_id", "target_dow", "target_month"]

COLORS = {"actual": "#1f2937", "model": "#2563eb", "baseline": "#f59e0b"}


# --------------------------------------------------------------------------- data
def load(data_dir):
    data_dir = Path(data_dir)
    sales = pd.read_csv(data_dir / "sales_daily.csv", parse_dates=["date"])
    meds = pd.read_csv(data_dir / "medicines.csv")
    meds["category_id"] = meds["category"].astype("category").cat.codes
    wide = sales.pivot(index="date", columns="medicine_id", values="units_sold").sort_index()
    promo = sales.pivot(index="date", columns="medicine_id", values="promo_flag").sort_index()
    return wide, promo, meds


def _trailing_mean(s, w):
    out = np.full(len(s), np.nan)
    c = np.cumsum(np.insert(s.astype(float), 0, 0.0))
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def build_rows(wide, promo, meds, mode):
    """mode='all'  -> every valid (origin, horizon) pair with a known target
       mode='last' -> only the final origin, with targets in the unseen future"""
    start = wide.index[0]
    n = len(wide)
    frames = []
    for mid in wide.columns:
        s = wide[mid].to_numpy(dtype=float)
        p = np.concatenate([promo[mid].to_numpy(dtype=float), np.zeros(HORIZON + 1)])
        cat = int(meds.loc[meds.medicine_id == mid, "category_id"].iloc[0])
        m7, m14, m28, m56 = (_trailing_mean(s, w) for w in (7, 14, 28, 56))
        sd28 = pd.Series(s).rolling(28).std().to_numpy()
        rm7c = pd.Series(s).rolling(7, center=True).mean().to_numpy()

        for h in range(1, HORIZON + 1):
            i = np.arange(WARMUP, n - h) if mode == "all" else np.array([n - 1])
            j = i + h
            k0 = int(np.ceil(h / 7))
            same_dow = np.mean([s[j - 7 * k] for k in range(k0, k0 + 4)], axis=0)
            ly_idx = j - 364
            ly_win = np.where(ly_idx >= 3, rm7c[np.clip(ly_idx, 0, n - 1)], np.nan)
            ly_lvl_idx = i - 364
            ly_lvl = np.where(ly_lvl_idx >= 27, m28[np.clip(ly_lvl_idx, 0, n - 1)], np.nan)
            tdate = start + pd.to_timedelta(j, unit="D")
            doy = tdate.dayofyear.to_numpy()
            frames.append(pd.DataFrame({
                "medicine_id": mid, "category_id": cat, "horizon": h,
                "origin_idx": i, "origin_date": start + pd.to_timedelta(i, unit="D"),
                "target_date": tdate,
                "target_dow": tdate.dayofweek.to_numpy(), "target_month": tdate.month.to_numpy(),
                "doy_sin": np.sin(2 * np.pi * doy / 365.25), "doy_cos": np.cos(2 * np.pi * doy / 365.25),
                "promo_target": p[j], "last1": s[i], "mean7": m7[i], "mean14": m14[i],
                "mean28": m28[i], "mean56": m56[i], "std28": sd28[i], "same_dow_mean": same_dow,
                "trend_ratio": m28[i] / (m56[i] + 0.5), "momentum_ratio": m7[i] / (m28[i] + 0.5),
                "ly_window": ly_win, "ly_ratio": ly_win / (ly_lvl + 0.5),
                "y": s[j] if mode == "all" else np.nan,
            }))
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------------ modelling
def make_model():
    return HistGradientBoostingRegressor(
        loss="poisson", learning_rate=0.06, max_iter=300, max_leaf_nodes=31,
        min_samples_leaf=100, l2_regularization=1.0, categorical_features=CATEGORICAL,
        early_stopping=False, random_state=0)


def metrics(y, pred):
    err = pred - y
    return {"MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "WAPE_%": float(100 * np.abs(err).sum() / y.sum()),
            "bias_%": float(100 * err.sum() / y.sum())}


def evaluate(df, model_pred):
    out = {"overall": {}, "by_horizon": {}, "by_medicine": {}}
    y = df["y"].to_numpy()
    preds = {"model": model_pred, "baseline_same_weekday": df["same_dow_mean"].to_numpy(),
             "baseline_moving_avg_28": df["mean28"].to_numpy()}
    for name, p in preds.items():
        out["overall"][name] = metrics(y, p)
    for label, lo, hi in [("1-7 days", 1, 7), ("8-14 days", 8, 14), ("15-28 days", 15, 28)]:
        m = (df["horizon"] >= lo) & (df["horizon"] <= hi)
        out["by_horizon"][label] = {n: metrics(y[m], p[m])["WAPE_%"] for n, p in preds.items()}
    for mid, g in df.groupby("medicine_name"):
        idx = g.index
        out["by_medicine"][mid] = {
            "model_WAPE_%": metrics(y[idx], model_pred[idx])["WAPE_%"],
            "baseline_WAPE_%": metrics(y[idx], preds["baseline_same_weekday"][idx])["WAPE_%"]}
    return out


# ---------------------------------------------------------------------------- plots
def plot_forecast_vs_actual(test, meds_to_plot, path):
    fig, axes = plt.subplots(len(meds_to_plot), 1, figsize=(10, 2.8 * len(meds_to_plot)), sharex=True)
    for ax, name in zip(np.atleast_1d(axes), meds_to_plot):
        g = test[(test.medicine_name == name) & (test.horizon == 14)].sort_values("target_date")
        roll = lambda c: g.set_index("target_date")[c].rolling(7, min_periods=1).mean()
        ax.plot(roll("y"), color=COLORS["actual"], lw=2, label="Actual")
        ax.plot(roll("pred"), color=COLORS["model"], lw=2, label="Model (made 14 days earlier)")
        ax.plot(roll("same_dow_mean"), color=COLORS["baseline"], lw=1.5, ls="--", label="Same-weekday baseline")
        ax.set_title(name, loc="left", fontsize=11, fontweight="bold")
        ax.set_ylabel("units/day (7-day avg)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.25)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.suptitle("Held-out period: forecast vs actual demand", fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.legend(handles, labels, frameon=False, ncol=3, fontsize=9, loc="lower center",
               bbox_to_anchor=(0.5, -0.03))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_error_by_horizon(test, path):
    rows = []
    for h, g in test.groupby("horizon"):
        y = g.y.to_numpy()
        rows.append((h, metrics(y, g.pred.to_numpy())["WAPE_%"],
                     metrics(y, g.same_dow_mean.to_numpy())["WAPE_%"],
                     metrics(y, g.mean28.to_numpy())["WAPE_%"]))
    r = pd.DataFrame(rows, columns=["h", "model", "same_dow", "ma28"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(r.h, r.model, color=COLORS["model"], lw=2.5, label="Gradient boosting model")
    ax.plot(r.h, r.same_dow, color=COLORS["baseline"], lw=2, ls="--", label="Same-weekday baseline")
    ax.plot(r.h, r.ma28, color="#9ca3af", lw=2, ls=":", label="28-day moving average")
    ax.set_xlabel("Forecast horizon (days ahead)")
    ax.set_ylabel("WAPE % (lower is better)")
    ax.set_title("Forecast error by horizon", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_importance(model, test, path, n=20000):
    sample = test.sample(min(n, len(test)), random_state=0)
    res = permutation_importance(model, sample[FEATURES], sample["y"], n_repeats=3,
                                 random_state=0, scoring="neg_mean_absolute_error")
    imp = pd.Series(res.importances_mean, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(imp.index, imp.values, color=COLORS["model"])
    ax.set_xlabel("Increase in MAE when feature is shuffled")
    ax.set_title("What drives the demand forecast", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return imp.sort_values(ascending=False).round(4).to_dict()


# ---------------------------------------------------------------------------- entry
def run(data_dir, out_dir, model_dir):
    out_dir, model_dir = Path(out_dir), Path(model_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    model_dir.mkdir(exist_ok=True, parents=True)

    wide, promo, meds = load(data_dir)
    names = meds.set_index("medicine_id")["medicine"]
    df = build_rows(wide, promo, meds, "all")
    df["medicine_name"] = df["medicine_id"].map(names)

    # chronological split: no training target after TRAIN_END, test origins after it
    train = df[(df.target_date <= TRAIN_END) & (df.origin_idx % 2 == 0)]
    test = df[df.origin_date > TRAIN_END].reset_index(drop=True)

    model = make_model().fit(train[FEATURES], train["y"])
    test["pred"] = np.clip(model.predict(test[FEATURES]), 0, None)

    results = evaluate(test, test["pred"].to_numpy())
    results["split"] = {"train_rows": int(len(train)), "test_rows": int(len(test)),
                        "train_targets_up_to": str(TRAIN_END.date()),
                        "test_forecast_origins": f"{test.origin_date.min().date()} to {test.origin_date.max().date()}"}

    plot_forecast_vs_actual(test, ["Paracetamol 500mg", "Cetirizine 10mg", "ORS sachet"],
                            out_dir / "forecast_vs_actual.png")
    plot_error_by_horizon(test, out_dir / "error_by_horizon.png")
    results["feature_importance"] = plot_importance(model, test, out_dir / "feature_importance.png")

    # final model: refit on all history, forecast the next 28 days
    full = df[df.origin_idx % 2 == 0]
    final = make_model().fit(full[FEATURES], full["y"])
    fut = build_rows(wide, promo, meds, "last")
    fut["medicine"] = fut["medicine_id"].map(names)
    fut["forecast_units"] = np.clip(final.predict(fut[FEATURES]), 0, None).round(2)
    forecast = fut[["target_date", "medicine_id", "medicine", "horizon", "forecast_units"]]\
        .rename(columns={"target_date": "date"})
    forecast.to_csv(out_dir / "forecast_next_28d.csv", index=False)

    import joblib
    joblib.dump(final, model_dir / "demand_model.joblib")
    (out_dir / "demand_metrics.json").write_text(json.dumps(results, indent=2))
    return results, forecast


if __name__ == "__main__":
    root = Path(__file__).parent
    res, fc = run(root / "data", root / "outputs", root / "models")
    print(json.dumps(res["overall"], indent=2))
