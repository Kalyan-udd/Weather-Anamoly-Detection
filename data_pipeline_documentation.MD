# SkyGuard AI — Data Pipeline Documentation

This document explains every step we take to turn raw Open-Meteo weather data
into a model-ready dataset, and *why* each step exists. Read this before
touching the pipeline code, so changes don't accidentally undo something that
was done for a specific reason.

---

## 1. Raw data

We pull hourly `temperature`, `pressure`, and `relative_humidity` per station
(currently Kolkata, 2016-01-01 to 2023-12-31) via a custom Open-Meteo API
wrapper. This data is clean — Open-Meteo doesn't contain sensor faults, so any
anomalies we want to detect have to be injected by us (see Step 4).

**Why we reindex to a complete hourly range before anything else:**
Real API extractions can have missing hours. If we don't make gaps explicit,
a gradient calculation (Step 3) could silently compute "change per hour"
across an actual 3-hour gap and produce a misleadingly large or small number.
We reindex to a full hourly `date_range` first so any gap becomes a visible
`NaN` row instead of a silent shift.

---

## 2. Climatology (the "what's normal here, right now" reference table)

**The problem this solves:** A single national threshold, or even a single
per-city average, doesn't work. Different regions (and different hours of the
day, different months) have very different natural variability. A desert
station's temperature can swing several degrees in an hour perfectly normally;
a coastal station's same swing would be a real anomaly. Comparing raw values
directly across time/place is not meaningful.

**What we do:** Group historical data by `(month, hour)` — 12 × 24 = 288
buckets — and compute the mean and standard deviation of each parameter
*within that specific bucket only*. "January, 3 AM" gets its own mean/std,
completely separate from "January, 2 PM" or "June, 3 AM."

**Critical rule: climatology must be built from CLEAN data, before any
anomalies are injected.** If we injected anomalies first and then computed
climatology on that already-corrupted data, the anomalies would leak into and
distort what counts as "expected normal" — weakening our own ability to
detect them later. Build climatology once from the clean historical data,
save it, and reuse it every time we score new data (real or
anomaly-injected).

This 288-row table is small and reusable — it doesn't need to be recomputed
from scratch every time; it's saved as an artifact and looked up at inference
time.

---

## 3. Two kinds of z-score — and why we need BOTH

Once we have climatology, we convert readings into z-scores:

```
z = (observed − expected_for_this_month_hour) / expected_std_for_this_month_hour
```

We compute this **twice** per parameter:

- **Raw value z-score** (`temperature_zscore`, etc.) — is this actual reading
  unusual for this place/time?
- **Gradient z-score** (`temperature_gradient_zscore`, etc.) — is the *change
  from the previous hour* unusual for this place/time? (Gradient = `value[t] -
  value[t-1]`, computed with `.diff()`.)

**Why both are needed, not just one:**

| Failure pattern | Caught by raw z-score? | Caught by gradient z-score? |
|---|---|---|
| Instant spike | Yes | Yes |
| Frozen/stuck value | Yes (value stays wrong) | No — gradient goes to ~0 once frozen |
| Step-change fault that persists | Yes (for the whole duration) | Only at the single transition moment |
| Slow calibration drift | Yes (residual keeps growing) | No — by design, per-hour change stays small |
| Sudden onset of any of the above | Weaker (level takes a moment to look unusual) | Yes — this is what gradient is best at |

Gradient z-score and raw z-score are complementary, not redundant. Dropping
either one loses real detection coverage — verified empirically (see Step 6).

**Why not just use a fixed threshold (e.g., "+5°C/hour is always bad")?**
Tested and rejected — it reopens the exact regional-variance problem
climatology solves. A swing that's abnormal on the coast can be an ordinary
desert evening. A flat national number can't serve both without either
missing real coastal faults or false-alarming on normal desert swings.

**Why not just use StandardScaler on the whole dataset instead of
climatology?**
Also tested and rejected. A single scaler fit across an entire year blends
day/night and seasonal cycles into one inflated "normal" spread, which hides
real anomalies that occur within a specific hour/season (a warm 3 AM reading
can look "normal" against a scaler that's also seen hot summer afternoons,
even though it's a real anomaly for that specific hour).

---

## 4. Anomaly injection (creates our labeled training/testing data)

Since real data has no anomalies, we simulate them on a **copy** of the clean
data (never overwrite the original). Each injected event also writes the
correct label for the rows it touches.

| Type | Real-world cause | How we simulate it |
|---|---|---|
| `spike` | Electrical interference, corrupted single reading | One random timestamp, value ± a large randomized jump (6-12 std devs) |
| `frozen` | Sensor physically stuck | Random start + duration (3-12 hrs), whole span overwritten with one repeated value |
| `comm_error` | Network/telemetry dropout, not a sensor problem | Random start + duration (2-5 hrs), value set to `NaN` — missing, not wrong |
| `fault` (drift) | Calibration slipping over time | Random start + longer duration (2-14 days), a small bias that *grows* linearly each hour, rather than an instant jump |

**Why magnitude/duration are randomized every time:** if every spike were
exactly the same size, the model could learn to recognize our specific
injector's "signature" rather than the general concept of a spike. Randomizing
within realistic ranges forces it to learn the general shape instead. This
matters because the hackathon organizers will inject anomalies with their own
(different) parameters for evaluation.

**Verified sanity check** (see `full_pipeline.py` output): mean absolute
temperature z-score for genuine rows ≈ 0.8, vs. ≈ 23 for spike/fault rows —
confirms the anomalies are actually statistically distinguishable, not just
present in the label column.

**Known current gap:** `comm_error` rows do NOT show up in temperature
z-score at all when a *different* column was the one set to NaN — the raw/
gradient z-score approach can't see missingness. Comm_error needs its own
explicit missing-value check (e.g., a binary "is NaN" flag or a rolling count
of recent NaNs), not a statistical threshold. **Not yet implemented — next
step.**

---

## 5. Cyclical time encoding

For models that need explicit time-awareness (notably an LSTM working on raw
sequences, which has no built-in sense of "what hour is it"), we encode hour
and month as sin/cos pairs rather than raw integers:

```python
df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
df['month_sin'] = np.sin(2*np.pi*df['month']/12)
df['month_cos'] = np.cos(2*np.pi*df['month']/12)
```

**Why not just feed raw hour/month numbers?** Hour 23 and hour 0 are one hour
apart in reality but 23 units apart as raw numbers — a model would wrongly
treat them as very different. Sin/cos wraps the cycle so adjacent times stay
numerically close. **Why both sin AND cos, not just one?** Sine alone can't
distinguish, e.g., hour 6 from hour 18 (same sine value at points in the
cycle) — cosine breaks that symmetry so every hour has a unique fingerprint.

Note: this is a per-row transformation, unrelated to the climatology
`groupby` — no bucket-grouping needed here, just a formula applied to each
row's own hour/month.

---

## 6. Order of operations (do not reorder without reading Step 2's warning)

1. Load clean raw data, reindex to complete hourly range.
2. Build climatology from the **clean** data. Save it.
3. Copy the clean data → inject anomalies + labels into the copy.
4. Compute raw + gradient z-scores on the **injected copy**, looking up the
   **clean** climatology from Step 2.
5. Add cyclical hour/month encoding.
6. (Not yet built) Add a comm_error/missingness feature.
7. Window into sequences for the LSTM, or leave as flat rows for a tree-based
   model (Random Forest / XGBoost).

## Files in this repo
- `build_climatology_features.py` — Steps 1, 2, 3(z-score half of it).
- `generate_and_inject.py` — Step 4 (injectors) + LSTM windowing/training.
- `full_pipeline.py` — Steps 1-5 wired together in the correct order, tested
  end-to-end on Kolkata-shaped data.

## Still open
- Comm_error missingness feature (Step 6 above).
- Final choice between tree-based model (Random Forest/XGBoost — recommended
  primary, better explainability via SHAP) vs. LSTM (explored as a secondary/
  comparison approach).
- Sensor health status logic (healthy/degrading/faulty) — aggregation over a
  rolling window of anomaly frequency, not a per-event mapping. Not yet coded.
- Real-time streaming simulation for the demo.