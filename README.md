# Continuous Glucose Monitoring Explorer

Streamlit dashboard for exploring continuous glucose monitoring (CGM) data for people with type II diabetes. Includes:
- Bundled synthetic CGM days (various scenarios)
- Multi-day comparison and overlay
- Event logging (meals, medication, exercise)
- Time-in-range analysis, event detection, and pattern finding
- PDF report export
- User-friendly feedback and plain-language summaries

## Biomedical Context

The app demonstrates core CGM analytics used in outpatient type II diabetes care: time in range, hypo/hyper excursions, smoothing noisy data, visualizing glucose traces, and finding recurring patterns. Upload your own CSV(s) or start with the provided samples.

## Quick Start

### Opening in GitHub Codespaces

1. Click "Code" in GitHub, choose "Codespaces", then "Create codespace on main".
2. After the container builds, open the integrated terminal.

### Running the Streamlit App

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

If you prefer not to activate the venv, run with full paths: `.venv/bin/streamlit run app.py`.

Open the printed `Local URL` in your browser.

## Usage Guide

### Usage Guide

- **Choose data:** Use the sidebar to select a bundled sample, upload your own CSV, or upload multiple days for comparison. All CSVs must have columns `timestamp` and `glucose_mg_dL`.
- **Multi-day comparison:** Overlay multiple days on a normalized 24-hour timescale. See per-day summaries and spot trends.
- **Event logging:** Enter meal, medication, and exercise times (12-hour format, e.g. `08:00 AM`). Markers appear on plots for context.
- **Adjust thresholds:** Set time-in-range limits (default 70–180 mg/dL) and smoothing window.
- **Inspect the plot:** View raw and smoothed glucose traces with hypo/hyper markers and threshold lines. Event markers are color-coded.
- **Review metrics:** Time in range, low, high percentages plus summary stats update instantly. Estimated A1C is shown.
- **Event table:** Detected hypo/hyper episodes are listed with start, end, duration, and average glucose.
- **Pattern detection:** Find times of day when highs or lows commonly occur.
- **PDF export:** Generate a plain-language PDF report for single-day analysis.

## Data

### Data

Several synthetic CGM scenarios are provided for testing:

- `data/cgm_sample.csv`: Well-controlled day with meal excursions and one short hypo event.
- `data/cgm_hypoglycemia.csv`: Day with multiple hypoglycemic episodes (overnight, early morning, pre-lunch, late afternoon).
- `data/cgm_high_variability.csv`: Day with large glucose swings and variable control.
- `data/cgm_well_controlled.csv`: Day with stable glucose and minimal excursions.
- `data/cgm_weekend.csv`: Weekend pattern with different meal times and variability.
- `data/cgm_exercise_day.csv`: Day with exercise events and their impact on glucose.
- `data/cgm_sick_day.csv`: Day with illness-related glucose changes.

All files use ISO 8601 timestamps and mg/dL units.

## Project Structure

## Project Structure

- `app.py` – Streamlit application (fully commented for clarity).
- `data/` – Synthetic CGM datasets for various scenarios.
- `requirements.txt` – Python dependencies.
- `main.lua` – Original LÖVE sample (unused by the CGM app).

## Features

- Single-day and multi-day CGM visualization
- Event logging (meals, medication, exercise) with color-coded markers
- Time-in-range, hypo/hyper event detection, and summary metrics
- Pattern detection for recurring highs/lows
- PDF export for single-day reports
- Multiple realistic synthetic CGM datasets
- User-friendly, plain-language feedback


