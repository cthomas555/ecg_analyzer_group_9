import pandas as pd  # Data manipulation
import numpy as np  # Numerical operations
import altair as alt  # Interactive charting
import streamlit as st  # Web app framework
from pathlib import Path  # File path handling
from datetime import datetime, time  # Date/time utilities
from io import BytesIO  # In-memory file for PDF export

st.set_page_config(page_title="CGM Explorer", layout="wide")  # Set Streamlit app config

DATA_PATH = Path(__file__).parent / "data" / "cgm_sample.csv"  # Default sample data path

def load_sample_data() -> pd.DataFrame:
    """Load bundled CGM sample data; generate an empty frame if missing."""
    # Try to load the default sample CSV, or return an empty DataFrame if missing
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    else:
        df = pd.DataFrame(columns=["timestamp", "glucose_mg_dL"])
    return df

@st.cache_data(show_spinner=False)
def load_uploaded(file) -> pd.DataFrame:
    """Load a user-uploaded CSV file."""
    return pd.read_csv(file, parse_dates=["timestamp"])

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and sort the CGM data for analysis."""
    df = df.copy()
    df = df.dropna(subset=["timestamp", "glucose_mg_dL"])
    df = df.sort_values("timestamp")
    df["glucose_mg_dL"] = df["glucose_mg_dL"].astype(float)
    return df

def smooth_series(df: pd.DataFrame, window_points: int) -> pd.Series:
    """Apply a moving average to smooth glucose readings."""
    if window_points <= 1:
        return df["glucose_mg_dL"]
    return df["glucose_mg_dL"].rolling(window_points, min_periods=1, center=True).mean()

def time_in_ranges(values: pd.Series, lower: float, upper: float) -> dict:
    """Calculate percent of readings in, below, and above target range."""
    total = len(values)
    if total == 0:
        return {"tir_pct": 0.0, "hypo_pct": 0.0, "hyper_pct": 0.0}
    tir = ((values >= lower) & (values <= upper)).mean() * 100
    hypo = (values < lower).mean() * 100
    hyper = (values > upper).mean() * 100
    return {"tir_pct": tir, "hypo_pct": hypo, "hyper_pct": hyper}

def detect_runs(values: pd.Series, condition_fn):
    """Find consecutive runs of values meeting a condition (e.g., hypo/hyper events)."""
    runs = []
    start_idx = None
    for i, val in enumerate(values):
        if condition_fn(val):
            if start_idx is None:
                start_idx = i
        elif start_idx is not None:
            runs.append((start_idx, i - 1))
            start_idx = None
    if start_idx is not None:
        runs.append((start_idx, len(values) - 1))
    return runs

def runs_to_frame(df: pd.DataFrame, runs, label: str, sample_minutes: int = 5) -> pd.DataFrame:
    """Convert detected runs to a DataFrame for display and analysis."""
    rows = []
    for start, end in runs:
        start_ts = df.iloc[start]["timestamp"]
        end_ts = df.iloc[end]["timestamp"]
        duration_min = (end - start + 1) * sample_minutes
        avg_val = df.iloc[start : end + 1]["glucose_mg_dL"].mean()
        rows.append({
            "type": label,
            "start": start_ts,
            "end": end_ts,
            "start_12h": start_ts.strftime("%I:%M %p"),
            "end_12h": end_ts.strftime("%I:%M %p"),
            "duration_min": duration_min,
            "avg_glucose": round(avg_val, 1),
        })
    return pd.DataFrame(rows)

def plot_glucose(df: pd.DataFrame, smoothed: pd.Series, lower: float, upper: float):
    """
    Plot glucose readings and highlight hypo/hyper events.
    - Shows raw and smoothed glucose lines
    - Adds threshold lines for hypo/hyper
    - Marks points below/above thresholds
    """
    plot_df = df[["timestamp", "glucose_mg_dL"]].copy()
    plot_df["smoothed"] = smoothed

    # Base chart: time on x-axis
    base = alt.Chart(plot_df).encode(
        x=alt.X("timestamp:T", title="Time"),
    )

    # Raw glucose line (blue)
    raw_line = base.mark_line(color="#4c78a8", opacity=0.5).encode(
        y=alt.Y("glucose_mg_dL:Q", title="Glucose (mg/dL)"),
        tooltip=["timestamp:T", "glucose_mg_dL:Q"],
    )

    # Smoothed glucose line (orange)
    smooth_line = base.mark_line(color="#f58518", strokeWidth=2).encode(
        y="smoothed:Q",
        tooltip=["timestamp:T", "smoothed:Q"],
    )

    # Dashed lines for hypo/hyper thresholds
    rules = alt.Chart(pd.DataFrame({"level": [lower, upper]})).mark_rule(strokeDash=[4, 4], color="#666").encode(
        y="level:Q"
    )

    # Mark points below/above thresholds
    hyp_points = plot_df[plot_df["glucose_mg_dL"] < lower]
    hyper_points = plot_df[plot_df["glucose_mg_dL"] > upper]

    hyp_mark = alt.Chart(hyp_points).mark_point(color="#e45756", size=40).encode(
        x="timestamp:T",
        y="glucose_mg_dL:Q",
        tooltip=["timestamp:T", "glucose_mg_dL:Q"],
    )
    hyper_mark = alt.Chart(hyper_points).mark_point(color="#72b7b2", size=40).encode(
        x="timestamp:T",
        y="glucose_mg_dL:Q",
        tooltip=["timestamp:T", "glucose_mg_dL:Q"],
    )

    # Combine all chart layers
    chart = (raw_line + smooth_line + rules + hyp_mark + hyper_mark).properties(height=420)
    st.altair_chart(chart, use_container_width=True)

def generate_pdf_report(df, tir, stats, estimated_a1c, hypo_runs, hyper_runs, lower, upper):
    """
    Generate a simple PDF report summarizing CGM metrics and events.
    - Includes key metrics, event counts, and plain-language recommendations
    - Uses reportlab for PDF creation
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib import colors
    except ImportError:
        return None
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    styles = getSampleStyleSheet()
    story = []
    
    # Title section
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor("#2c3e50"))
    story.append(Paragraph("CGM Analysis Report", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Report date
    report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    story.append(Paragraph(f"<b>Report Generated:</b> {report_date}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Key metrics table
    story.append(Paragraph("<b>Key Metrics</b>", styles['Heading2']))
    metrics_data = [
        ["Time in Range", f"{tir['tir_pct']:.1f}%"],
        ["Time High", f"{tir['hyper_pct']:.1f}%"],
        ["Time Low", f"{tir['hypo_pct']:.1f}%"],
        ["Mean Glucose", f"{stats['Mean']} mg/dL"],
        ["Estimated A1C", f"{estimated_a1c:.1f}%"],
        ["Glucose Range", f"{stats['Min']} - {stats['Max']} mg/dL"],
    ]
    metrics_table = Table(metrics_data, colWidths=[2.5*inch, 2*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Event summary section
    story.append(Paragraph("<b>Event Summary</b>", styles['Heading2']))
    story.append(Paragraph(f"Hypoglycemic events: <b>{len(hypo_runs)}</b> (below {lower} mg/dL)", styles['Normal']))
    story.append(Paragraph(f"Hyperglycemic events: <b>{len(hyper_runs)}</b> (above {upper} mg/dL)", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Recommendations in plain language
    story.append(Paragraph("<b>Plain Language Summary</b>", styles['Heading2']))
    recommendations = []
    if tir["tir_pct"] >= 70:
        recommendations.append("Most readings stay within your target range — maintain your current plan.")
    else:
        recommendations.append("Time in range is below 70%; bring this to your care team to review meds, meals, or activity.")
    
    if len(hyper_runs) > 0:
        recommendations.append(f"High episodes detected: {len(hyper_runs)}. Note meal/med timing; follow clinician guidance.")
    
    if len(hypo_runs) > 0:
        recommendations.append(f"Low episodes detected: {len(hypo_runs)}. Treat per your hypoglycemia plan and notify care team if repeating.")
    
    if stats["Std dev"] > 30:
        recommendations.append("Glucose swings are wide (higher variability); routines or timing may be shifting.")
    
    for rec in recommendations:
        story.append(Paragraph(f"• {rec}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Medical disclaimer
    story.append(Paragraph("<i>This report is for informational purposes only and is not medical advice. Always follow your clinician's guidance.</i>", styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def main():
    # Main entry point for the Streamlit app
    st.title("Continuous Glucose Monitoring Explorer")
    st.write(
        "Analyze and visualize continuous glucose monitoring (CGM) data for type II diabetes. "
        "Upload your own CSV or explore the bundled sample day at 5-minute resolution."
    )
    st.caption("For people already diagnosed with type II diabetes. Not medical advice; follow your clinician's plan.")

    # --- Sidebar: Data selection and event logging ---
    st.sidebar.header("Dataset")
    # User chooses between sample, single upload, or multi-day upload
    source = st.sidebar.radio("Choose data", ["Sample data", "Upload CSV", "Upload Multiple Days"], index=0)

    st.sidebar.header("Event Logging")
    st.sidebar.write("Track meals, medication, and exercise to see their impact on glucose.")

    # Meals: user enters meal times and picks marker color
    with st.sidebar.expander("🍽️ Meals", expanded=False):
        meal_times = st.text_area(
            "Enter meal times (HH:MM AM/PM, one per line)",
            placeholder="07:30 AM\n12:00 PM\n06:30 PM",
            height=80,
            key="meals",
        )
        meal_color = st.color_picker("Meal marker color", value="#ff6b6b", key="meal_color")

    # Medication: user enters medication times and picks marker color
    with st.sidebar.expander("💊 Medication", expanded=False):
        med_times = st.text_area(
            "Enter medication times (HH:MM AM/PM, one per line)",
            placeholder="08:00 AM\n08:00 PM",
            height=60,
            key="meds",
        )
        med_color = st.color_picker("Medication marker color", value="#4CAF50", key="med_color")

    # Exercise: user enters exercise times and picks marker color
    with st.sidebar.expander("🏃 Exercise", expanded=False):
        exercise_times = st.text_area(
            "Enter exercise times (HH:MM AM/PM, one per line)",
            placeholder="06:00 AM\n05:00 PM",
            height=60,
            key="exercise",
        )
        exercise_color = st.color_picker("Exercise marker color", value="#2196F3", key="exercise_color")

    # --- Data upload logic ---
    uploaded_file = None
    uploaded_files = None
    if source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload CSV with columns timestamp, glucose_mg_dL", type=["csv"])
        if uploaded_file is None:
            st.info("Upload a CSV to proceed or switch to the sample dataset.")
    elif source == "Upload Multiple Days":
        uploaded_files = st.sidebar.file_uploader(
            "Upload multiple CSV files (one per day)",
            type=["csv"],
            accept_multiple_files=True,
        )
        if not uploaded_files:
            st.info("Upload CSV files to compare multiple days or switch to sample dataset.")

    # --- Data loading and preparation ---
    df = pd.DataFrame()
    multi_day_mode = False
    all_days = []

    # Load data based on user selection
    if source == "Sample data":
        df = load_sample_data()
    elif source == "Upload Multiple Days" and uploaded_files:
        multi_day_mode = True
        for file in uploaded_files:
            try:
                day_df = load_uploaded(file)
                day_df = prepare(day_df)
                day_df["day_label"] = file.name  # Label for each day
                all_days.append(day_df)
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Could not read {file.name}: {exc}")
        if all_days:
            df = pd.concat(all_days, ignore_index=True)
        else:
            st.warning("No valid CSV files loaded.")
            return
    elif uploaded_file is not None:
        try:
            df = load_uploaded(uploaded_file)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read file: {exc}")
            return

    # --- Data validation ---
    if df.empty:
        st.warning("No data available. Ensure the CSV has columns timestamp and glucose_mg_dL.")
        return

    if not multi_day_mode:
        df = prepare(df)

    # --- Analysis settings ---
    default_low, default_high = 70, 180
    # User sets hypo/hyper thresholds
    lower, upper = st.sidebar.slider(
        "Time-in-range limits (mg/dL)",
        min_value=50,
        max_value=250,
        value=(default_low, default_high),
        step=5,
    )
    # Smoothing window for moving average
    smooth_minutes = st.sidebar.slider("Smoothing window (minutes)", 0, 60, 15, step=5)
    # Sampling period (resolution)
    sample_minutes = st.sidebar.selectbox("Sampling period (minutes)", [1, 5, 15], index=1)

    # Resample data if user changes sampling period
    if sample_minutes != 5:
        df = df.set_index("timestamp").resample(f"{sample_minutes}min").mean(numeric_only=True).interpolate()
        df = df.reset_index()

    window_points = max(1, int(smooth_minutes / sample_minutes))
    smoothed = smooth_series(df, window_points)

    # --- Parse event times from user input (12-hour format) ---
    def parse_times_12hr(time_input):
        """Parse times like '08:00 AM' from user input."""
        times = []
        if time_input.strip():
            for line in time_input.strip().split("\n"):
                line = line.strip()
                if line:
                    try:
                        # Parse 12-hour format like "08:00 AM" or "06:30 PM"
                        time_parts = line.upper().split()
                        if len(time_parts) == 2:
                            time_str, period = time_parts
                            hour, minute = map(int, time_str.split(":"))
                            # Convert to 24-hour format
                            if period == "PM" and hour != 12:
                                hour += 12
                            elif period == "AM" and hour == 12:
                                hour = 0
                            times.append(time(hour, minute))
                    except (ValueError, IndexError):
                        pass
        return times

    # Get event marker times for each category
    meal_markers = parse_times_12hr(meal_times)
    med_markers = parse_times_12hr(med_times)
    exercise_markers = parse_times_12hr(exercise_times)

    # Multi-day comparison visualization
    if multi_day_mode:
        # --- Multi-day comparison section ---
        st.subheader("Multi-Day Comparison")
        st.caption("Compare glucose patterns across multiple days to spot trends.")

        # Normalize all days to same time-of-day scale (ignore dates)
        normalized_days = []
        for day_df in all_days:
            day_df_copy = day_df.copy()
            # Extract time of day and create normalized datetime on same base date
            day_df_copy["time_of_day"] = day_df_copy["timestamp"].dt.time
            base_date = datetime(2025, 1, 1)
            day_df_copy["normalized_time"] = day_df_copy["timestamp"].apply(
                lambda x: datetime.combine(base_date.date(), x.time())
            )
            normalized_days.append(day_df_copy)

        df_normalized = pd.concat(normalized_days, ignore_index=True)

        # Overlay all days on same 24-hour timescale
        comparison_chart = alt.Chart(df_normalized).mark_line(strokeWidth=2).encode(
            x=alt.X("normalized_time:T", title="Time of Day", axis=alt.Axis(format="%I:%M %p")),
            y=alt.Y("glucose_mg_dL:Q", title="Glucose (mg/dL)", scale=alt.Scale(domain=[50, 300])),
            color=alt.Color("day_label:N", legend=alt.Legend(title="Day")),
            tooltip=["day_label:N", "time_of_day:N", "glucose_mg_dL:Q"],
        )

        # Add dashed threshold lines for hypo/hyper
        rules = alt.Chart(pd.DataFrame({"level": [lower, upper]})).mark_rule(
            strokeDash=[4, 4], 
            color="#666"
        ).encode(y="level:Q")

        # Add event markers for meals, meds, exercise
        base_date = datetime(2025, 1, 1)
        marker_layers = []

        if meal_markers:
            meal_times_norm = [datetime.combine(base_date.date(), t) for t in meal_markers]
            meal_df = pd.DataFrame({
                "time": meal_times_norm,
                "label": ["🍽️ Meal"] * len(meal_times_norm),
            })
            meal_chart = alt.Chart(meal_df).mark_rule(strokeDash=[6, 3], strokeWidth=2).encode(
                x="time:T",
                color=alt.value(meal_color),
                tooltip=["label:N", "time:T"]
            )
            marker_layers.append(meal_chart)

        if med_markers:
            med_times_norm = [datetime.combine(base_date.date(), t) for t in med_markers]
            med_df = pd.DataFrame({
                "time": med_times_norm,
                "label": ["💊 Medication"] * len(med_times_norm),
            })
            med_chart = alt.Chart(med_df).mark_rule(strokeDash=[6, 3], strokeWidth=2).encode(
                x="time:T",
                color=alt.value(med_color),
                tooltip=["label:N", "time:T"]
            )
            marker_layers.append(med_chart)

        if exercise_markers:
            exercise_times_norm = [datetime.combine(base_date.date(), t) for t in exercise_markers]
            exercise_df = pd.DataFrame({
                "time": exercise_times_norm,
                "label": ["🏃 Exercise"] * len(exercise_times_norm),
            })
            exercise_chart = alt.Chart(exercise_df).mark_rule(strokeDash=[6, 3], strokeWidth=2).encode(
                x="time:T",
                color=alt.value(exercise_color),
                tooltip=["label:N", "time:T"]
            )
            marker_layers.append(exercise_chart)

        # Combine all chart layers
        combined = comparison_chart + rules
        for layer in marker_layers:
            combined = combined + layer
        combined = combined.properties(height=450)
        st.altair_chart(combined, use_container_width=True)

        # --- Per-day summary table ---
        st.subheader("Daily Summaries")
        daily_summaries = []
        for day_df in all_days:
            tir_day = time_in_ranges(day_df["glucose_mg_dL"], lower, upper)
            daily_summaries.append({
                "Day": day_df["day_label"].iloc[0],
                "Mean (mg/dL)": round(day_df["glucose_mg_dL"].mean(), 1),
                "Time in Range (%)": round(tir_day["tir_pct"], 1),
                "Low (%)": round(tir_day["hypo_pct"], 1),
                "High (%)": round(tir_day["hyper_pct"], 1),
            })
        st.dataframe(pd.DataFrame(daily_summaries), use_container_width=True)
        st.info("Look for patterns: Are weekends different? Did changes in medication or routine improve control?")
        return

    # --- Single-day summary and event detection ---
    # Calculate metrics before layout (needed for PDF export)
    tir = time_in_ranges(df["glucose_mg_dL"], lower, upper)  # Time in range, hypo, hyper
    hypo_runs = detect_runs(df["glucose_mg_dL"], lambda v: v < lower)  # Find low events
    hyper_runs = detect_runs(df["glucose_mg_dL"], lambda v: v > upper)  # Find high events
    stats = {
        "Mean": round(df["glucose_mg_dL"].mean(), 1),
        "Median": round(df["glucose_mg_dL"].median(), 1),
        "Std dev": round(df["glucose_mg_dL"].std(), 1),
        "Min": round(df["glucose_mg_dL"].min(), 1),
        "Max": round(df["glucose_mg_dL"].max(), 1),
    }
    estimated_a1c = (stats["Mean"] + 46.7) / 28.7  # ADA eAG formula

    left_col, right_col = st.columns([2, 1])
    with left_col:
        # --- Main glucose plot ---
        plot_glucose(df, smoothed, lower, upper)
        # Overlay event markers for meals, meds, and exercise if provided
        if meal_markers or med_markers or exercise_markers:
            # Replot with event markers
            plot_df = df[["timestamp", "glucose_mg_dL"]].copy()
            plot_df["smoothed"] = smoothed
            base = alt.Chart(plot_df).encode(x=alt.X("timestamp:T", title="Time"))
            raw_line = base.mark_line(color="#4c78a8", opacity=0.5).encode(
                y=alt.Y("glucose_mg_dL:Q", title="Glucose (mg/dL)"),
                tooltip=["timestamp:T", "glucose_mg_dL:Q"],
            )
            smooth_line = base.mark_line(color="#f58518", strokeWidth=2).encode(
                y="smoothed:Q",
                tooltip=["timestamp:T", "smoothed:Q"],
            )
            rules = alt.Chart(pd.DataFrame({"level": [lower, upper]})).mark_rule(strokeDash=[4, 4], color="#666").encode(y="level:Q")
            hyp_points = plot_df[plot_df["glucose_mg_dL"] < lower]
            hyper_points = plot_df[plot_df["glucose_mg_dL"] > upper]
            hyp_mark = alt.Chart(hyp_points).mark_point(color="#e45756", size=40).encode(
                x="timestamp:T", y="glucose_mg_dL:Q", tooltip=["timestamp:T", "glucose_mg_dL:Q"]
            )
            hyper_mark = alt.Chart(hyper_points).mark_point(color="#72b7b2", size=40).encode(
                x="timestamp:T", y="glucose_mg_dL:Q", tooltip=["timestamp:T", "glucose_mg_dL:Q"]
            )

            # Create marker charts for each event category
            marker_layers = []
            if meal_markers:
                meal_df = pd.DataFrame({
                    "time": [df["timestamp"].iloc[0].replace(hour=t.hour, minute=t.minute) for t in meal_markers],
                    "label": ["🍽️ Meal"] * len(meal_markers),
                })
                meal_chart = alt.Chart(meal_df).mark_rule(strokeDash=[6, 3], strokeWidth=2).encode(
                    x="time:T",
                    color=alt.value(meal_color),
                    tooltip=["label:N", "time:T"]
                )
                marker_layers.append(meal_chart)

            if med_markers:
                med_df = pd.DataFrame({
                    "time": [df["timestamp"].iloc[0].replace(hour=t.hour, minute=t.minute) for t in med_markers],
                    "label": ["💊 Medication"] * len(med_markers),
                })
                med_chart = alt.Chart(med_df).mark_rule(strokeDash=[6, 3], strokeWidth=2).encode(
                    x="time:T",
                    color=alt.value(med_color),
                    tooltip=["label:N", "time:T"]
                )
                marker_layers.append(med_chart)

            if exercise_markers:
                exercise_df = pd.DataFrame({
                    "time": [df["timestamp"].iloc[0].replace(hour=t.hour, minute=t.minute) for t in exercise_markers],
                    "label": ["🏃 Exercise"] * len(exercise_markers),
                })
                exercise_chart = alt.Chart(exercise_df).mark_rule(strokeDash=[6, 3], strokeWidth=2).encode(
                    x="time:T",
                    color=alt.value(exercise_color),
                    tooltip=["label:N", "time:T"]
                )
                marker_layers.append(exercise_chart)

            # Combine all chart layers
            chart = raw_line + smooth_line + rules + hyp_mark + hyper_mark
            for layer in marker_layers:
                chart = chart + layer
            chart = chart.properties(height=420)
            st.altair_chart(chart, use_container_width=True)
    with right_col:
        # --- Summary metrics and plain-language feedback ---
        st.subheader("Time in Range")
        st.metric("In range (%)", f"{tir['tir_pct']:.1f}")
        st.metric("High (%)", f"{tir['hyper_pct']:.1f}")
        st.metric("Low (%)", f"{tir['hypo_pct']:.1f}")

        # Plain-language guidance for non-technical users
        st.subheader("Plain Language Summary")
        alert_level = "good"
        summary_lines = []
        if tir["hypo_pct"] > 0 or len(hypo_runs) > 0:
            alert_level = "high"
            summary_lines.append("Lows detected below your set limit. Follow your hypoglycemia plan (treat, recheck, notify care team if recurring).")
        if tir["hyper_pct"] >= 20 or len(hyper_runs) > 0:
            alert_level = "medium" if alert_level != "high" else alert_level
            summary_lines.append("High readings are common; review meal timing, meds, or activity per your care team's guidance.")
        if tir["tir_pct"] < 70:
            alert_level = "medium" if alert_level != "high" else alert_level
            summary_lines.append("Time in range is below 70%. Your clinician may adjust medication, meals, or activity to raise time in range.")
        if not summary_lines:
            summary_lines.append("Glucose is mostly within your chosen range. Keep monitoring and follow your usual plan.")

        # Show alert based on severity
        joined = " ".join(summary_lines)
        if alert_level == "high":
            st.error(joined)
        elif alert_level == "medium":
            st.warning(joined)
        else:
            st.success(joined)

        # Show summary metrics
        st.subheader("Summary")
        top = st.columns(3)
        bottom = st.columns(2)
        top[0].metric("Mean (mg/dL)", stats["Mean"])
        top[1].metric("Median (mg/dL)", stats["Median"])
        top[2].metric("Std dev", stats["Std dev"])
        bottom[0].metric("Min (mg/dL)", stats["Min"])
        bottom[1].metric("Max (mg/dL)", stats["Max"])

        # A1C estimation
        st.subheader("Estimated A1C")
        st.metric("eA1C (%)", f"{estimated_a1c:.1f}")
        st.caption("Estimated A1C from average glucose (eAG formula). Not a substitute for lab A1C.")
        if estimated_a1c < 7.0:
            st.success("Below ADA target of 7% for many adults with type 2 diabetes.")
        elif estimated_a1c < 8.0:
            st.info("Near common targets; discuss with your care team.")
        else:
            st.warning("Above 8%; consider discussing treatment adjustments with your clinician.")

        # Meaning-focused interpretation for non-technical users
        st.subheader("What this means today")
        highs = len(hyper_runs)
        lows = len(hypo_runs)
        variability = stats["Std dev"]
        interpretation = []
        if tir["tir_pct"] >= 70:
            interpretation.append("Most readings stay within your target range — maintain your current plan.")
        else:
            interpretation.append("Time in range is below 70%; bring this to your care team to review meds, meals, or activity.")

        if highs > 0:
            interpretation.append(f"High episodes: {highs}. Note meal/med timing at those hours; follow your clinician's guidance on corrections.")
        else:
            interpretation.append("No high episodes detected with the current limits.")

        if lows > 0:
            interpretation.append(f"Low episodes: {lows}. Treat per your hypoglycemia plan and notify your care team if repeating.")
        else:
            interpretation.append("No low episodes detected with the current limits.")

        if variability > 30:
            interpretation.append("Glucose swings are wide today (higher variability); routines or timing may be shifting.")
        else:
            interpretation.append("Glucose swings are moderate; day-to-day patterns look fairly steady.")

        st.markdown("\n".join(f"- {line}" for line in interpretation))

    st.divider()

    # --- Event detection: show detected hypo/hyper events ---
    st.subheader("Event Detection")
    events_df = pd.concat([
        runs_to_frame(df, hypo_runs, "hypo", sample_minutes),
        runs_to_frame(df, hyper_runs, "hyper", sample_minutes),
    ])
    st.caption("Hypo: below lower limit. Hyper: above upper limit. Adjust the limits in the sidebar if prescribed differently.")
    if not events_df.empty:
        hyp_count = (events_df["type"] == "hypo").sum()
        hyper_count = (events_df["type"] == "hyper").sum()
        longest = int(events_df["duration_min"].max())
        m1, m2, m3 = st.columns(3)
        m1.metric("Hypo events", hyp_count)
        m2.metric("Hyper events", hyper_count)
        m3.metric("Longest event (min)", longest)

        # Timeline bar chart for events
        timeline = alt.Chart(events_df).mark_bar(height=12).encode(
            x=alt.X("start:T", title="Time", axis=alt.Axis(format="%I:%M %p")),
            x2="end:T",
            y=alt.Y("type:N", title=""),
            color=alt.Color(
                "type:N",
                scale=alt.Scale(domain=["hypo", "hyper"], range=["#e45756", "#72b7b2"]),
                legend=alt.Legend(title="Event"),
            ),
            tooltip=["type", "start_12h:N", "end_12h:N", "duration_min:Q", "avg_glucose:Q"],
        ).properties(height=90)
        st.altair_chart(timeline, use_container_width=True)

        # Show event details in a table
        display_df = events_df[["type", "start_12h", "end_12h", "duration_min", "avg_glucose"]].copy()
        display_df.columns = ["Type", "Start", "End", "Duration (min)", "Avg Glucose (mg/dL)"]
        st.dataframe(display_df.reset_index(drop=True), use_container_width=True)
    else:
        st.write("No hypo or hyper events detected with current thresholds.")

    # --- Pattern detection for recurring event times ---
    st.divider()
    st.subheader("Pattern Detection")
    st.caption("Find times of day when highs or lows commonly occur.")

    if len(hypo_runs) > 0 or len(hyper_runs) > 0:
        from collections import Counter
        hypo_hours = [df.iloc[start]["timestamp"].hour for start, _ in hypo_runs]
        hyper_hours = [df.iloc[start]["timestamp"].hour for start, _ in hyper_runs]

        pattern_text = []
        if hypo_hours:
            hypo_counter = Counter(hypo_hours)
            most_common_hypo = hypo_counter.most_common(1)[0]
            hour_12h = f"{most_common_hypo[0] % 12 or 12}:00 {'AM' if most_common_hypo[0] < 12 else 'PM'}"
            pattern_text.append(f"**Lows** occur most often around **{hour_12h}** ({most_common_hypo[1]} time(s)).")

        if hyper_hours:
            hyper_counter = Counter(hyper_hours)
            most_common_hyper = hyper_counter.most_common(1)[0]
            hour_12h = f"{most_common_hyper[0] % 12 or 12}:00 {'AM' if most_common_hyper[0] < 12 else 'PM'}"
            pattern_text.append(f"**Highs** occur most often around **{hour_12h}** ({most_common_hyper[1]} time(s)).")

        st.markdown("\\n\\n".join(pattern_text))
        st.info("Consider adjusting meals, medication timing, or activity around these hours with your clinician.")
    else:
        st.write("No patterns detected—glucose stays within range throughout the day.")

    st.subheader("Data Preview")
    st.dataframe(df.head(200), use_container_width=True)

    st.sidebar.download_button(
        label="Download sample CSV",
        data=load_sample_data().to_csv(index=False),
        file_name="cgm_sample.csv",
        mime="text/csv",
    )
    
    # PDF Export (only for single-day view)
    if not multi_day_mode:
        st.sidebar.divider()
        st.sidebar.header("Export Report")
        if st.sidebar.button("Generate PDF Report"):
            pdf_buffer = generate_pdf_report(df, tir, stats, estimated_a1c, hypo_runs, hyper_runs, lower, upper)
            if pdf_buffer:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.sidebar.download_button(
                    label="Download PDF",
                    data=pdf_buffer,
                    file_name=f"cgm_report_{timestamp}.pdf",
                    mime="application/pdf",
                )
                st.sidebar.success("Report ready to download!")
            else:
                st.sidebar.error("PDF generation requires reportlab. Install with: pip install reportlab")

if __name__ == "__main__":
    main()
