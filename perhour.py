# rowcount.py — robust parser + hourly deviation charts for muon .log files
# Time from col 6 (or 6–10 if split). Counts from cols 1,3,4,5.
# Adds hourly aggregation and hourly deviation (%) charts.

from __future__ import annotations
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Muon Log Dashboard", layout="wide")
st.title("Muon Detector Log Dashboard")
st.caption("Upload a .log file. Time from column 6 (or 6–10 if split). Counts from columns 1, 3, 4, 5. Hourly deviations included.")

# ---------------- Sidebar ----------------
st.sidebar.header("Upload")
file = st.sidebar.file_uploader("Choose a .log/.txt/.csv", type=["log", "txt", "csv"])

st.sidebar.header("Parsing")
sep_mode = st.sidebar.selectbox(
    "Separator",
    ["Spaces (1+ spaces)", "Comma (,)", "Tab (\\t)", "Auto (comma / tab / spaces)"],
    index=0,
    help="Pick the exact separator your file uses. Prefer explicit options over Auto."
)
has_header = st.sidebar.checkbox("File has a header row", value=False)
skip_rows = st.sidebar.number_input("Skip first N lines", min_value=0, value=0, step=1)

st.sidebar.header("Column mapping (1-based)")
time_col_1b = st.sidebar.number_input("Time column (start)", min_value=1, value=6, step=1)
c1_1b = st.sidebar.number_input("Column 1 — Total (per minute)",  min_value=1, value=1, step=1)
c3_1b = st.sidebar.number_input("Column 3 — Top + Middle",        min_value=1, value=3, step=1)
c4_1b = st.sidebar.number_input("Column 4 — Top + Bottom",        min_value=1, value=4, step=1)
c5_1b = st.sidebar.number_input("Column 5 — Middle + Bottom",     min_value=1, value=5, step=1)

timestamp_split_5 = st.sidebar.checkbox(
    "Timestamp is split across 5 columns (Fri | Aug | 23 | 17:02:07 | 2024)",
    value=True,
)

st.sidebar.header("Time format")
custom_fmt = st.sidebar.text_input(
    "Custom datetime format",
    value="%a %b %d %H:%M:%S %Y",
)

st.sidebar.header("Performance")
max_points = st.sidebar.slider("Max points per chart (minute-level)", 1_000, 200_000, 10_000, step=1_000)

st.sidebar.header("Hourly aggregation")
agg_choice = st.sidebar.selectbox("Aggregate per hour using", ["Mean", "Sum", "Median"], index=0)
dev_mode = st.sidebar.selectbox(
    "Deviation calculation",
    [
        "Hourly mean vs overall mean (default)",
        "Average minute deviation within hour",
    ],
    index=0,
    help="Default: compare each hour's aggregate to the overall aggregate mean. "
         "Alternative: average of per-minute deviations within each hour."
)
dev_threshold = st.sidebar.slider("Flag deviations over (%)", 0.0, 200.0, 10.0, step=0.5)

def sep_and_engine(mode: str):
    if "Spaces" in mode: return r"[ ]+", "python"
    if "Comma"  in mode: return ",", "c"
    if "Tab"    in mode: return "\t", "python"
    return r"[,\t ]+", "python"   # Auto

sep, engine = sep_and_engine(sep_mode)

@st.cache_data(show_spinner=False)
def load_df(_file, sep, engine, header_flag, skip_rows):
    # Read as strings; only pass low_memory for the C engine
    kwargs = dict(sep=sep, engine=engine, dtype=str, skiprows=skip_rows, on_bad_lines="skip")
    if engine == "c":
        kwargs["low_memory"] = False
    return pd.read_csv(_file, header=(0 if header_flag else None), **kwargs)

def clean_numeric(s: pd.Series) -> pd.Series:
    # Robust: strip spaces, remove thousands separators, drop textual Nones, keep digits/. - and exponent
    s2 = s.astype(str).str.strip()
    s2 = s2.str.replace(",", "", regex=False)
    s2 = s2.str.replace(r"\b(None|nan|NaN)\b", "", regex=True)
    s2 = s2.str.replace(r"[^0-9.\-eE]", "", regex=True)
    return pd.to_numeric(s2, errors="coerce")

def to_datetime_joined(df: pd.DataFrame, start_idx: int, fmt: str, split5: bool) -> pd.Series:
    if split5:
        parts = [df.iloc[:, start_idx + k].astype(str).str.strip() for k in range(5)]
        txt = parts[0] + " " + parts[1] + " " + parts[2] + " " + parts[3] + " " + parts[4]
    else:
        txt = df.iloc[:, start_idx].astype(str).str.strip()
    return pd.to_datetime(txt, format=fmt, errors="coerce")

# ---------------- Main ----------------
if not file:
    st.info("⬅️ Upload a .log file to begin.")
    st.stop()

df = load_df(file, sep=sep, engine=engine, header_flag=has_header, skip_rows=skip_rows)

with st.expander("Raw file peek (first 5 rows, first 12 columns)"):
    st.dataframe(df.iloc[:5, :min(12, df.shape[1])], use_container_width=True)

# 1-based → 0-based indexes
t0 = time_col_1b - 1
i1, i3, i4, i5 = c1_1b - 1, c3_1b - 1, c4_1b - 1, c5_1b - 1

max_needed = max(i1, i3, i4, i5, t0 if not timestamp_split_5 else t0 + 4)
if df.shape[1] <= max_needed:
    st.error(f"File has {df.shape[1]} columns; you referenced index {max_needed}. "
             f"Check separator/mapping/split toggle, or increase 'Skip first N lines'.")
    st.stop()

# Minute-level parse
time = to_datetime_joined(df, t0, custom_fmt, timestamp_split_5)
if time.isna().all():
    with st.expander("Preview of raw time values"):
        cols = list(range(t0, t0 + (5 if timestamp_split_5 else 1)))
        st.dataframe(df.iloc[:10, cols], use_container_width=True)
    st.error("Time parse failed. Verify separator, 'split 5' toggle, and the format (%a %b %d %H:%M:%S %Y).")
    st.stop()

total     = clean_numeric(df.iloc[:, i1])
top_mid   = clean_numeric(df.iloc[:, i3])
top_bot   = clean_numeric(df.iloc[:, i4])
mid_bot   = clean_numeric(df.iloc[:, i5])

work = pd.DataFrame({
    "time":             time,
    "Total (per min)":  total,
    "Top + Middle":     top_mid,
    "Top + Bottom":     top_bot,
    "Middle + Bottom":  mid_bot,
}).dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

# Minute-level diagnostics
valid_counts = {
    "Total (per min)":  work["Total (per min)"].notna().sum(),
    "Top + Middle":     work["Top + Middle"].notna().sum(),
    "Top + Bottom":     work["Top + Bottom"].notna().sum(),
    "Middle + Bottom":  work["Middle + Bottom"].notna().sum(),
}
st.info(
    f"Rows (minute-level) after time-parse: {len(work):,} | "
    f"Valid — Total:{valid_counts['Total (per min)']:,}, "
    f"T+M:{valid_counts['Top + Middle']:,}, "
    f"T+B:{valid_counts['Top + Bottom']:,}, "
    f"M+B:{valid_counts['Middle + Bottom']:,}"
)

# ---------------- Hourly aggregation ----------------
agg_map = {"Mean": "mean", "Sum": "sum", "Median": "median"}
agg_func = agg_map[agg_choice]

hourly = (
    work.set_index("time")
        .resample("H")
        .agg({
            "Total (per min)": agg_func,
            "Top + Middle":    agg_func,
            "Top + Bottom":    agg_func,
            "Middle + Bottom": agg_func,
        })
        .dropna(how="all")
)

# ---------------- Hourly deviation calculations ----------------
coinc_cols = ["Top + Middle", "Top + Bottom", "Middle + Bottom"]

# Method 1: Hourly aggregate vs overall mean of hourly aggregates
hourly_dev_vs_overall = pd.DataFrame(index=hourly.index)
for c in coinc_cols:
    mean_overall = hourly[c].mean(skipna=True)
    if pd.notna(mean_overall) and mean_overall != 0:
        hourly_dev_vs_overall[f"{c} Δ% (hourly vs overall)"] = (hourly[c].sub(mean_overall).abs() / mean_overall) * 100.0
    else:
        hourly_dev_vs_overall[f"{c} Δ% (hourly vs overall)"] = pd.NA

# Method 2: Average minute deviation within each hour
# dev_minute = mean over minutes in the hour of |x - hour_mean| / hour_mean * 100
grouped = work.set_index("time").groupby(pd.Grouper(freq="H"))
hourly_avg_minute_dev = pd.DataFrame(index=hourly.index)
for c in coinc_cols:
    # Hourly mean for each minute row (aligned)
    hour_means = grouped[c].transform("mean")
    # Avoid division by zero
    denom = hour_means.where(hour_means != 0)
    dev_minute = (work.set_index("time")[c].sub(hour_means).abs() / denom) * 100.0
    hourly_avg_minute_dev[f"{c} Δ% (avg minute dev)"] = dev_minute.groupby(pd.Grouper(freq="H")).mean()

# ---------------- Downsample (minute-level only) ----------------
n = len(work)
stride = max(1, n // max_points)
plot_min = work.iloc[::stride].copy()

# ---------------- UI: Tabs ----------------
tabs = st.tabs(["Counts (minute)", "Counts (hourly)", "Hourly deviation (%)"])

# ---- Counts (minute) ----
with tabs[0]:
    c1col, c2col = st.columns(2)
    with c1col:
        st.subheader("Total (per minute)")
        st.line_chart(plot_min.set_index("time")[["Total (per min)"]])
    with c2col:
        st.subheader("Top + Middle (per minute)")
        st.line_chart(plot_min.set_index("time")[["Top + Middle"]])

    c3col, c4col = st.columns(2)
    with c3col:
        st.subheader("Top + Bottom (per minute)")
        st.line_chart(plot_min.set_index("time")[["Top + Bottom"]])
    with c4col:
        st.subheader("Middle + Bottom (per minute)")
        st.line_chart(plot_min.set_index("time")[["Middle + Bottom"]])

# ---- Counts (hourly) ----
with tabs[1]:
    st.caption(f"Hourly aggregation: **{agg_choice}**")
    h1, h2 = st.columns(2)
    with h1:
        st.subheader("Top + Middle (hourly)")
        st.line_chart(hourly[["Top + Middle"]])
    with h2:
        st.subheader("Top + Bottom (hourly)")
        st.line_chart(hourly[["Top + Bottom"]])
    h3 = st.container()
    with h3:
        st.subheader("Middle + Bottom (hourly)")
        st.line_chart(hourly[["Middle + Bottom"]])

# ---- Hourly deviation (%) ----
with tabs[2]:
    st.caption(
        "Choose deviation method in the sidebar. "
        "• *Hourly mean vs overall mean*: deviation of each hour’s aggregate from the overall aggregate mean. "
        "• *Average minute deviation within hour*: average of per-minute deviations relative to that hour’s mean."
    )

    if dev_mode.startswith("Hourly mean"):
        df_dev = hourly_dev_vs_overall
        lbl = "Δ% (hourly vs overall mean)"
    else:
        df_dev = hourly_avg_minute_dev
        lbl = "Δ% (average minute deviation within hour)"

    d1, d2 = st.columns(2)
    with d1:
        st.subheader(f"Top + Middle — {lbl}")
        st.line_chart(df_dev[[col for col in df_dev.columns if col.startswith('Top + Middle')]])
    with d2:
        st.subheader(f"Top + Bottom — {lbl}")
        st.line_chart(df_dev[[col for col in df_dev.columns if col.startswith('Top + Bottom')]])

    d3 = st.container()
    with d3:
        st.subheader(f"Middle + Bottom — {lbl}")
        st.line_chart(df_dev[[col for col in df_dev.columns if col.startswith('Middle + Bottom')]])

    # Flag large deviations
    over = pd.DataFrame(index=df_dev.index)
    for c in coinc_cols:
        colname = [col for col in df_dev.columns if col.startswith(c)][0]
        over[c] = df_dev[colname]
    flagged = over[(over > dev_threshold).any(axis=1)].dropna(how="all")

    st.markdown(f"**Hours exceeding {dev_threshold:.1f}% deviation**")
    st.dataframe(flagged.head(500), use_container_width=True)

# ---- Download ----
# Include hourly aggregates and deviation tables for offline analysis
with st.expander("Downloads"):
    st.download_button(
        "Download minute-level cleaned CSV",
        data=work.to_csv(index=False).encode(),
        file_name="muon_cleaned_minute.csv",
        mime="text/csv",
    )
    hcsv = hourly.reset_index().rename(columns={"index": "time"})
    st.download_button(
        "Download hourly aggregates CSV",
        data=hcsv.to_csv(index=False).encode(),
        file_name="muon_hourly.csv",
        mime="text/csv",
    )
    if dev_mode.startswith("Hourly mean"):
        dcsv = df_dev.reset_index().rename(columns={"index": "time"})
    else:
        dcsv = df_dev.reset_index().rename(columns={"index": "time"})
    st.download_button(
        "Download hourly deviation CSV",
        data=dcsv.to_csv(index=False).encode(),
        file_name="muon_hourly_deviation.csv",
        mime="text/csv",
    )

