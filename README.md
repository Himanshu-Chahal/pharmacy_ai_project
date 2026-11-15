# Pharmacy Medicine Demand & Expiry-Risk Forecasting

**Domain:** AI & Machine Learning  
**Goal:** Help a pharmacy order the right amount of each medicine, and warn it early about batches that will expire unsold.

The project answers two linked questions:

1. **Demand forecasting** – how many units of each medicine will we sell each day for the next 28 days?
2. **Expiry risk** – of the batches on the shelf today, which will *not* be sold before their expiry date, how many units, and what is that worth?

## Quick start

```bash
pip install -r requirements.txt
python run_pipeline.py        # ~25 seconds, regenerates everything
```

Outputs land in `outputs/`:

| File | What it is |
|---|---|
| `forecast_next_28d.csv` | Daily unit forecast per medicine for the next 28 days |
| `expiry_risk_report.csv` | Every live batch with risk probability, projected units expiring, value at risk, tier and recommended action |
| `demand_metrics.json`, `expiry_metrics.json` | All evaluation numbers |
| `forecast_vs_actual.png`, `error_by_horizon.png`, `feature_importance.png`, `expiry_risk_overview.png` | Charts |

## Project layout

```
data_generation.py    synthetic sales + batch-level inventory simulator
demand_forecasting.py multi-horizon gradient-boosting demand model
expiry_risk.py        expiry classifier + forecast-based projection + risk report
run_pipeline.py       runs all three and prints a summary
```

## Data

Real pharmacy data is private, so `data_generation.py` simulates 26 medicines over 2024-01-01 → 2026-10-31 (26,910 daily rows). It includes:

- weekday pattern (busy Mondays, quiet Sundays), yearly growth, count noise
- seasonality by illness family (flu, allergy, gastro, summer) plus random outbreaks
- planned promotions
- a realistic inventory layer: weekly order-up-to replenishment, pack sizes, remaining shelf life on delivery, first-expiry-first-out (FEFO) dispensing, occasional supplier bulk orders, and expiry write-offs

To use real data instead, supply the same CSV columns (`sales_daily.csv`: date, medicine_id, units_sold, promo_flag; `batches.csv` and `expiry_snapshots.csv` as produced by the generator) and skip the generation step.

## Method

### 1. Demand forecasting (`demand_forecasting.py`)
One **global** `HistGradientBoostingRegressor` (Poisson loss, suited to count data) is trained for all medicines at once using *direct multi-horizon* forecasting: each row is (medicine, forecast date, horizon 1–28 days) and the target is units sold on that future day.

Features: recent level and trend (last day, 7/14/28/56-day means, 28-day std), same-weekday average, **sales around the same date last year**, calendar of the target day, planned-promotion flag, medicine and category.

Validation is strictly chronological. The model is trained only on targets up to 2026-07-31 and scored on forecasts issued 2026-08-01 → 2026-10-30. It is compared with two baselines.

### 2. Expiry risk (`expiry_risk.py`)
Two views are combined:

- **Classifier.** Every week each live batch is snapshotted (days to expiry, units left, units sitting in earlier-expiring batches that will be dispensed first, recent sales rate, days of stock cover). The label is whether the batch eventually expired with units unsold. Monotonic constraints ensure risk can only rise as days of cover rises. Trained on outcomes known by 2026-03-31, tested on snapshots after.
- **Forecast projection.** The 28-day demand forecast is consumed FEFO up to each batch's expiry date; whatever is left is the expected number of units to expire, valued at unit cost.

A batch is **High** risk if either view is alarmed (probability ≥ 0.6 or ≥ 50% of the batch projected to expire), **Medium** at ≥ 0.3 / ≥ 20%, otherwise **Low**, each with a suggested action (discount, transfer, return to supplier, pause reorders, or just dispense first).

## Results (held-out data)

**Demand forecast (WAPE = total absolute error ÷ total units; lower is better)**

| Model | WAPE | MAE (units/day/medicine) |
|---|---|---|
| **Gradient boosting** | **24.6%** | **4.48** |
| 28-day moving average | 27.5% | 5.02 |
| Same-weekday (seasonal naive) | 28.6% | 5.22 |

The model wins at every horizon (1–7, 8–14, 15–28 days) and for all 26 medicines, with the biggest gains on seasonal items (loperamide −16.6 pts, cough syrup −10.3, oseltamivir −10.2). Error stays flat across horizons (about 24–25%), most likely because the model uses last-year seasonality as well as recent sales.

**Expiry risk**

| | ROC-AUC | PR-AUC |
|---|---|---|
| **Classifier** | **0.989** | **0.950** |
| Days-of-cover rule (baseline) | 0.978 | 0.880 |

At a 0.5 threshold: precision 0.88, recall 0.85. Applied to the 82 live batches at 2026-10-31: **8 High, 2 Medium, 72 Low**, with about **INR 45,700** of stock projected to expire (Salbutamol inhaler, Amox-clav syrup and Loperamide batches lead the list).

## Limitations — please read before quoting these numbers

- **The data is synthetic.** The numbers show the pipeline works, not how it would perform on a real pharmacy. Real data would be messier (stockouts hide true demand, returns, price changes, local events).
- **Daily demand is mostly noise.** A 24.6% WAPE is near what count noise allows for items selling 5–40 units a day; do not expect single-digit error at daily level. Aggregating to weekly forecasts is much more accurate.
- **Expiry test set is small:** only 25 expired batches in the test period, so treat AUC/precision as indicative. The classifier beats the simple days-of-cover rule only modestly because, in this simulation, expiry is driven mostly by stock cover. The learned model's main value is combining signals; on real data with richer drivers the gap may be larger or smaller.
- **Test selection bias:** batches still unsold at the end of the data have no known outcome and are excluded, which inflates the test positive rate (16% vs 6% in training).
- **Waste is exaggerated on purpose** (simulated write-offs are roughly 19% of cost of goods sold) so there are enough expired batches to learn from; real pharmacy write-off is typically a few percent.
- **The two risk views can disagree** (e.g. a nearly-empty batch queued behind large earlier batches may get a low classifier probability but a high projected loss). The tier takes the more alarmed view and both numbers are in the report.
- Forecasts beyond 28 days assume the average level continues.

## Ideas for extending it

- Add prediction intervals and turn them into a recommended order quantity (safety stock).
- Replace the weekly reorder rule with the model's forecast and measure the simulated drop in waste and stockouts.
- Add local holidays, weather and disease-surveillance signals as features.
- Wrap `forecast_next_28d.csv` and `expiry_risk_report.csv` in a Streamlit dashboard.
