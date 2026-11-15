"""
Synthetic pharmacy data generator.

Simulates ~3 years of daily sales for a mid-sized pharmacy together with the
batch-level inventory behind those sales (FEFO dispensing, weekly reordering,
expiry write-offs). Real pharmacy data is private, so this gives the project
realistic structure to learn from:

  * weekly pattern (busy Mondays, quiet Sundays)
  * disease / weather seasonality (flu, allergy, gastro, summer)
  * random outbreaks, promotions, yearly growth, count noise
  * order-up-to replenishment with pack sizes and short shelf-life items,
    which is what creates expiry waste

Outputs (in ./data):
  medicines.csv          catalogue
  sales_daily.csv        one row per medicine per day
  batches.csv            one row per received batch with its final outcome
  expiry_snapshots.csv   weekly state of every live batch (for the expiry model)
  current_stock.csv      state of every live batch on the last day
"""
from pathlib import Path

import numpy as np
import pandas as pd

START = "2024-01-01"
END = "2026-10-31"
SEED = 42
LEAD_TIME = 2        # days from order to delivery
REVIEW_PERIOD = 7    # order every 7 days
WEEKLY = np.array([1.12, 1.06, 1.0, 1.0, 1.0, 0.95, 0.78])  # Mon..Sun

# base = average units/day, shelf = shelf life (days) at manufacture,
# cover = days of stock the (imperfect) reorder policy aims to hold
CATALOG = [
    dict(name="Paracetamol 500mg",     category="Analgesic",    base=42,  shelf=1095, season=("flu", 0.9),     price=1.8,  cost=1.2,  pack=100, cover=10, growth=0.06),
    dict(name="Azithromycin 500mg",    category="Antibiotic",   base=14,  shelf=730,  season=("flu", 0.8),     price=18,   cost=12,   pack=30,  cover=14, growth=0.04),
    dict(name="Amoxicillin 500mg",     category="Antibiotic",   base=16,  shelf=730,  season=("flu", 0.6),     price=9,    cost=6,    pack=30,  cover=14, growth=0.04),
    dict(name="Cetirizine 10mg",       category="Antiallergic", base=24,  shelf=1095, season=("allergy", 1.0), price=2.5,  cost=1.6,  pack=100, cover=12, growth=0.05),
    dict(name="Pantoprazole 40mg",     category="Gastro",       base=28,  shelf=730,  season=("gastro", 0.3),  price=6,    cost=4,    pack=60,  cover=14, growth=0.07),
    dict(name="Metformin 500mg",       category="Diabetes",     base=30,  shelf=1095, season=("none", 0),      price=1.5,  cost=1.0,  pack=100, cover=21, growth=0.08),
    dict(name="Amlodipine 5mg",        category="Cardio",       base=26,  shelf=1095, season=("none", 0),      price=2,    cost=1.2,  pack=100, cover=21, growth=0.07),
    dict(name="Atorvastatin 10mg",     category="Cardio",       base=20,  shelf=1095, season=("none", 0),      price=5,    cost=3.2,  pack=60,  cover=21, growth=0.07),
    dict(name="ORS sachet",            category="Rehydration",  base=12,  shelf=540,  season=("summer", 1.6),  price=20,   cost=14,   pack=50,  cover=10, growth=0.03),
    dict(name="Cough syrup 100ml",     category="Respiratory",  base=10,  shelf=240,  season=("flu", 1.4),     price=85,   cost=58,   pack=24,  cover=45, growth=0.02),
    dict(name="Vitamin D3 60k",        category="Supplement",   base=8,   shelf=365,  season=("none", 0),      price=35,   cost=22,   pack=20,  cover=60, growth=0.10),
    dict(name="Multivitamin",          category="Supplement",   base=14,  shelf=730,  season=("flu", 0.2),     price=7,    cost=4.5,  pack=60,  cover=20, growth=0.05),
    dict(name="Ibuprofen 400mg",       category="Analgesic",    base=18,  shelf=1095, season=("flu", 0.4),     price=2.2,  cost=1.4,  pack=100, cover=14, growth=0.03),
    dict(name="Omeprazole 20mg",       category="Gastro",       base=22,  shelf=730,  season=("gastro", 0.3),  price=3,    cost=2,    pack=100, cover=14, growth=0.05),
    dict(name="Insulin glargine pen",  category="Diabetes",     base=1.6, shelf=120,  season=("none", 0),      price=620,  cost=480,  pack=12,  cover=70, growth=0.08),
    dict(name="Salbutamol inhaler",    category="Respiratory",  base=3,   shelf=180,  season=("allergy", 0.8), price=180,  cost=130,  pack=10,  cover=70, growth=0.03),
    dict(name="Antacid gel 170ml",     category="Gastro",       base=9,   shelf=300,  season=("gastro", 0.5),  price=110,  cost=75,   pack=12,  cover=50, growth=0.02),
    dict(name="Lubricant eye drops",   category="Ophthalmic",   base=6,   shelf=200,  season=("summer", 0.6),  price=140,  cost=90,   pack=12,  cover=70, growth=0.04),
    dict(name="Diclofenac gel",        category="Analgesic",    base=8,   shelf=365,  season=("none", 0),      price=95,   cost=60,   pack=20,  cover=50, growth=0.03),
    dict(name="Levothyroxine 50mcg",   category="Endocrine",    base=18,  shelf=730,  season=("none", 0),      price=2.8,  cost=1.8,  pack=100, cover=21, growth=0.06),
    dict(name="Loperamide 2mg",        category="Gastro",       base=5,   shelf=365,  season=("gastro", 1.0),  price=2.5,  cost=1.5,  pack=60,  cover=60, growth=0.02),
    dict(name="Montelukast 10mg",      category="Respiratory",  base=13,  shelf=730,  season=("allergy", 0.8), price=10,   cost=6.5,  pack=30,  cover=14, growth=0.05),
    dict(name="Povidone iodine 100ml", category="Antiseptic",   base=7,   shelf=365,  season=("none", 0),      price=75,   cost=50,   pack=24,  cover=60, growth=0.02),
    dict(name="Oseltamivir 75mg",      category="Antiviral",    base=2.0, shelf=270,  season=("flu", 2.5),     price=55,   cost=38,   pack=10,  cover=90, growth=0.02),
    dict(name="Amox-clav syrup",       category="Antibiotic",   base=4,   shelf=150,  season=("flu", 0.8),     price=110,  cost=75,   pack=12,  cover=60, growth=0.03),
    dict(name="Ondansetron 4mg",       category="Gastro",       base=6,   shelf=730,  season=("gastro", 0.6),  price=5,    cost=3,    pack=30,  cover=20, growth=0.03),
]


def _bump(doy, centre, width):
    d = np.abs(doy - centre)
    d = np.minimum(d, 365 - d)
    return np.exp(-0.5 * (d / width) ** 2)


SEASON_SHAPE = {
    "flu": lambda doy: _bump(doy, 15, 35) + 0.45 * _bump(doy, 225, 25),
    "allergy": lambda doy: _bump(doy, 75, 25) + 0.6 * _bump(doy, 290, 20),
    "gastro": lambda doy: _bump(doy, 210, 30) + 0.35 * _bump(doy, 140, 20),
    "summer": lambda doy: _bump(doy, 135, 35),
    "none": lambda doy: np.zeros_like(doy, dtype=float),
}


def _new_batch(bid, med_id, t, qty, life):
    return dict(id=bid, med=med_id, recv=t, exp=t + life, qty=qty, rem=qty,
                expired=0, resolved=None, shelf=life)


def _snapshot(t, active, sales, dates, med_id):
    live = sorted((b for b in active if b["rem"] > 0 and b["exp"] > t),
                  key=lambda b: (b["exp"], b["id"]))
    r7 = sales[max(0, t - 6): t + 1].mean()
    r28 = sales[max(0, t - 27): t + 1].mean()
    total = sum(b["rem"] for b in live)
    rows, ahead = [], 0
    for b in live:
        rows.append(dict(
            batch_id=b["id"], medicine_id=med_id, snapshot_date=dates[t],
            days_to_expiry=b["exp"] - t, units_remaining=b["rem"],
            units_received=b["qty"], batch_age_days=t - b["recv"],
            units_ahead=ahead, total_on_hand=total, rate7=r7, rate28=r28,
            shelf_life_days=b["shelf"]))
        ahead += b["rem"]
    return rows


def generate(out_dir="data", seed=SEED):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    dates = pd.date_range(START, END, freq="D")
    n = len(dates)
    dow, doy = dates.dayofweek.values, dates.dayofyear.values

    # shared outbreak events per illness family
    events = {}
    for kind in ["flu", "gastro", "allergy"]:
        mult = np.ones(n)
        for _ in range(4):
            s = rng.integers(0, n - 12)
            mult[s:s + 10] *= rng.uniform(1.25, 1.5)
        events[kind] = mult

    sales_rows, batch_rows, snap_rows, current_rows, med_rows = [], [], [], [], []
    batch_id = 0

    for med_id, spec in enumerate(CATALOG):
        kind, amp = spec["season"]
        season = 1 + amp * SEASON_SHAPE[kind](doy)
        trend = (1 + spec["growth"]) ** (np.arange(n) / 365)
        event = events.get(kind, np.ones(n))

        # planned promotions (known in advance, so usable as a feature)
        promo = np.zeros(n, dtype=int)
        for s in np.flatnonzero(rng.random(n) < 0.012):
            promo[s:s + 7] = 1
        promo_mult = np.where(promo == 1, rng.uniform(1.2, 1.5), 1.0)

        lam = spec["base"] * WEEKLY[dow] * season * event * trend * promo_mult
        demand = rng.poisson(lam * rng.gamma(30, 1 / 30, n))

        pack, cover, shelf = spec["pack"], spec["cover"], spec["shelf"]
        sales = np.zeros(n, dtype=int)
        stockout = np.zeros(n, dtype=int)
        expired_daily = np.zeros(n, dtype=int)
        onhand = np.zeros(n, dtype=int)
        all_b, active, pending = [], [], []

        # opening stock
        qty0 = int(np.ceil(spec["base"] * (cover + 9) / pack) * pack)
        b0 = _new_batch(batch_id, med_id, 0, qty0, int(shelf * rng.uniform(0.5, 1.0)))
        batch_id += 1
        active.append(b0)
        all_b.append(b0)
        offset = med_id % REVIEW_PERIOD

        for t in range(n):
            # 1. deliveries
            still = []
            for arr, qty, life in pending:
                if arr == t:
                    b = _new_batch(batch_id, med_id, t, qty, life)
                    batch_id += 1
                    active.append(b)
                    all_b.append(b)
                else:
                    still.append((arr, qty, life))
            pending = still

            # 2. write off anything that reached its expiry date
            for b in active:
                if b["exp"] <= t and b["rem"] > 0:
                    b["expired"] = b["rem"]
                    expired_daily[t] += b["rem"]
                    b["rem"] = 0
                    b["resolved"] = t

            # 3. dispense first-expiry-first-out
            active.sort(key=lambda b: (b["exp"], b["id"]))
            need, sold = int(demand[t]), 0
            for b in active:
                if need == 0:
                    break
                take = min(b["rem"], need)
                b["rem"] -= take
                need -= take
                sold += take
                if b["rem"] == 0 and b["resolved"] is None:
                    b["resolved"] = t
            sales[t], stockout[t] = sold, need
            active = [b for b in active if b["rem"] > 0]
            onhand[t] = sum(b["rem"] for b in active)

            # 4. weekly snapshot of live batches
            if t >= 56 and t % 7 == 0:
                snap_rows += _snapshot(t, active, sales, dates, med_id)

            # 5. weekly order-up-to replenishment (moving-average based)
            if t % REVIEW_PERIOD == offset:
                ma28 = sales[max(0, t - 27): t + 1].mean() if t > 0 else spec["base"]
                target = ma28 * rng.uniform(0.6, 1.15) * (LEAD_TIME + REVIEW_PERIOD + cover)
                position = onhand[t] + sum(q for _, q, _ in pending)
                q = target - position
                if q > 0:
                    if rng.random() < 0.08:          # supplier bulk scheme -> overstock
                        q = max(q, target * rng.uniform(1.0, 2.0))
                    q = int(np.ceil(q / pack) * pack)
                    life = int(shelf * rng.uniform(0.2, 1.0))  # remaining shelf life on arrival
                    pending.append((t + LEAD_TIME, q, life))

        current_rows += [dict(r, is_current=1) for r in _snapshot(n - 1, active, sales, dates, med_id)]

        for b in all_b:
            sold_b = b["qty"] - b["rem"] - b["expired"]
            status = "expired" if b["expired"] > 0 else ("open" if b["rem"] > 0 else "sold_out")
            batch_rows.append(dict(
                batch_id=b["id"], medicine_id=med_id,
                received_date=dates[b["recv"]], expiry_date=dates[0] + pd.Timedelta(days=b["exp"]),
                qty_received=b["qty"], qty_sold=sold_b, qty_expired=b["expired"],
                qty_remaining_end=b["rem"], status=status,
                resolved_date=(dates[b["resolved"]] if b["resolved"] is not None else pd.NaT),
                shelf_life_days_at_receipt=b["shelf"]))

        for t in range(n):
            sales_rows.append((dates[t], med_id, int(sales[t]), int(promo[t]),
                               int(stockout[t]), int(onhand[t]), int(expired_daily[t])))
        med_rows.append(dict(medicine_id=med_id, medicine=spec["name"], category=spec["category"],
                             unit_price=spec["price"], unit_cost=spec["cost"], pack_size=pack,
                             shelf_life_days=shelf))

    meds = pd.DataFrame(med_rows)
    sales_df = pd.DataFrame(sales_rows, columns=["date", "medicine_id", "units_sold", "promo_flag",
                                                 "stockout_units", "stock_on_hand_eod", "units_expired"])
    sales_df = sales_df.merge(meds[["medicine_id", "medicine", "category"]], on="medicine_id")

    meds.to_csv(out / "medicines.csv", index=False)
    sales_df.to_csv(out / "sales_daily.csv", index=False)
    pd.DataFrame(batch_rows).to_csv(out / "batches.csv", index=False)
    pd.DataFrame(snap_rows).to_csv(out / "expiry_snapshots.csv", index=False)
    pd.DataFrame(current_rows).to_csv(out / "current_stock.csv", index=False)
    return meds, sales_df


if __name__ == "__main__":
    meds, sales = generate(Path(__file__).parent / "data")
    print(f"{len(meds)} medicines, {len(sales):,} daily rows, "
          f"{sales['date'].min().date()} -> {sales['date'].max().date()}")
