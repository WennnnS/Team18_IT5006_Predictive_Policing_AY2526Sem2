import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import date

from utils import (
    load_test_data,
    load_train_data,
    filter_single_record,
    filter_batch_records,
)
from inference import predict_rows


# Basic page config. Wide layout is more suitable here because the app
# contains tables, metrics, and a ranking chart on the same page.
st.set_page_config(
    page_title="Chicago Crime Hotspot Prediction",
    layout="wide",
)

# A little CSS helps a lot for demo readability.
# The default Streamlit sidebar is too narrow and the radio text is too small,
# so we enlarge the sidebar and make the navigation easier to read.
# The risk badges are also styled here because plain text was not visually strong enough.
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            min-width: 340px !important;
            max-width: 340px !important;
        }

        section[data-testid="stSidebar"] .stMarkdown h2,
        section[data-testid="stSidebar"] .stMarkdown h3,
        section[data-testid="stSidebar"] .stMarkdown p {
            color: #1f2937 !important;
        }

        section[data-testid="stSidebar"] .stMarkdown h2 {
            font-size: 2rem !important;
            font-weight: 800 !important;
            margin-bottom: 0.3rem !important;
        }

        section[data-testid="stSidebar"] .stCaption,
        section[data-testid="stSidebar"] small {
            font-size: 1.05rem !important;
            color: #6b7280 !important;
        }

        section[data-testid="stSidebar"] label[data-baseweb="radio"] {
            font-size: 1.18rem !important;
            font-weight: 700 !important;
            padding-top: 0.15rem !important;
            padding-bottom: 0.15rem !important;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] > label {
            margin-bottom: 0.55rem !important;
        }

        .risk-badge-high {
            display: inline-block;
            padding: 0.45rem 0.9rem;
            border-radius: 999px;
            background-color: #fee2e2;
            color: #b91c1c;
            font-weight: 800;
            font-size: 1.15rem;
            border: 1px solid #fecaca;
            margin: 0.35rem 0 0.85rem 0;
        }

        .risk-badge-low {
            display: inline-block;
            padding: 0.45rem 0.9rem;
            border-radius: 999px;
            background-color: #dcfce7;
            color: #166534;
            font-weight: 800;
            font-size: 1.15rem;
            border: 1px solid #bbf7d0;
            margin: 0.35rem 0 0.85rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# The model uses 4-hour time blocks.
# Showing raw block numbers alone is not user-friendly, so we map them to actual time windows.
TIME_BLOCK_LABELS = {
    0: "00:00–03:59",
    1: "04:00–07:59",
    2: "08:00–11:59",
    3: "12:00–15:59",
    4: "16:00–19:59",
    5: "20:00–23:59",
}


def format_district(d):
    # The data stores districts as numbers. For display, "District 4" is much clearer than plain "4".
    try:
        return f"District {int(d)}"
    except Exception:
        return f"District {d}"


def format_time_block(tb):
    # This turns block codes into something people can actually read during demo.
    label = TIME_BLOCK_LABELS.get(tb, "Unknown")
    return f"Block {tb} ({label})"


def hotspot_text(v):
    # Keep the official model terminology here, since the task is still binary hotspot classification.
    if pd.isna(v):
        return "N/A"
    return "Hotspot" if int(v) == 1 else "Not hotspot"


def risk_text(v):
    # This is just a simpler UI layer for presentation.
    # The model is still binary underneath.
    if pd.isna(v):
        return "N/A"
    return "High Risk" if int(v) == 1 else "Low Risk"


def render_prediction_banner(prob, pred_label, threshold):
    # The banner is the first thing the audience should notice.
    # Red means predicted hotspot, green means predicted not hotspot.
    if pred_label == 1:
        st.error(
            f"High Risk. Predicted hotspot with probability {prob:.3f}, above threshold {threshold:.2f}."
        )
    else:
        st.success(
            f"Low Risk. Predicted not hotspot with probability {prob:.3f}, below threshold {threshold:.2f}."
        )


def build_display_sample_row(row: pd.Series) -> pd.DataFrame:
    # The raw row contains many columns. For presentation, only keep the fields
    # that help explain what this record looks like and why the prediction is meaningful.
    display_dict = {
        "District": format_district(row.get("district")),
        "Date": str(pd.to_datetime(row.get("event_date")).date()) if pd.notna(row.get("event_date")) else "N/A",
        "Time Window": format_time_block(row.get("time_block")),
        "Crime Count": row.get("crime_count", "N/A"),
        "Current Hotspot Status": hotspot_text(row.get("hotspot_current")) if "hotspot_current" in row.index else "N/A",
        "Rolling Mean (6 blocks)": row.get("rolling_mean_6_blocks", "N/A"),
        "Rolling Mean (42 blocks)": row.get("rolling_mean_42_blocks", "N/A"),
        "Actual Next-Block Label": hotspot_text(row.get("target_hotspot_next_block"))
        if "target_hotspot_next_block" in row.index else "N/A",
    }
    return pd.DataFrame([display_dict])


def prepare_risk_scan_table(result: pd.DataFrame) -> pd.DataFrame:
    # This table is used for the ranking view in Daily Risk Scan.
    # The point is not to dump every raw feature, but to show a clean summary of the top records.
    out = result.copy()

    out["District"] = out["district"].apply(format_district)
    out["Time Window"] = out["time_block"].apply(format_time_block)
    out["Date"] = pd.to_datetime(out["event_date"], errors="coerce").dt.date.astype(str)
    out["Predicted Label"] = out["predicted_label"].apply(hotspot_text)
    out["Risk Level"] = out["predicted_label"].apply(risk_text)

    if "target_hotspot_next_block" in out.columns:
        out["Actual Label"] = out["target_hotspot_next_block"].apply(hotspot_text)
    else:
        out["Actual Label"] = "N/A"

    # Highest probability first, because the scan page is about prioritization.
    out = out.sort_values("predicted_probability", ascending=False).reset_index(drop=True)
    out["Rank"] = out.index + 1

    display_cols = [
        "Rank",
        "District",
        "Date",
        "Time Window",
        "predicted_probability",
        "Risk Level",
        "Predicted Label",
        "Actual Label",
    ]

    rename_map = {
        "predicted_probability": "Predicted Probability",
    }

    return out[display_cols].rename(columns=rename_map)


def get_date_bounds(df: pd.DataFrame):
    # The app only demos on the 2025 test set, so the date picker should stay within that range.
    dates = pd.to_datetime(df["event_date"], errors="coerce").dt.date.dropna()
    return dates.min(), dates.max()


def get_manual_options(df: pd.DataFrame):
    # These are the selectable values used in manual mode.
    districts = sorted(df["district"].dropna().unique().tolist())
    time_blocks = sorted(df["time_block"].dropna().unique().tolist())
    return districts, time_blocks


def compute_daily_metrics(result: pd.DataFrame):
    # These are light summary stats for the selected day.
    # I keep them simple because this page is mainly about ranking and scanning,
    # not full evaluation reporting.
    metrics = {}
    metrics["rows_scanned"] = len(result)
    metrics["predicted_hotspots"] = int((result["predicted_label"] == 1).sum())
    metrics["highest_probability"] = float(result["predicted_probability"].max()) if len(result) else 0.0
    metrics["average_probability"] = float(result["predicted_probability"].mean()) if len(result) else 0.0

    if "target_hotspot_next_block" in result.columns:
        y_true = result["target_hotspot_next_block"].astype(int)
        y_pred = result["predicted_label"].astype(int)

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        actual_hotspots = int((y_true == 1).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        metrics["actual_hotspots"] = actual_hotspots
        metrics["correct_hotspot_alerts"] = tp
        metrics["daily_precision"] = precision
        metrics["daily_recall"] = recall
    else:
        metrics["actual_hotspots"] = None
        metrics["correct_hotspot_alerts"] = None
        metrics["daily_precision"] = None
        metrics["daily_recall"] = None

    return metrics


# utils.py already handles file existence checks and basic validation,
# so here we mostly assume the loading step is safe.
df_test = load_test_data()
df_train = load_train_data()

min_date, max_date = get_date_bounds(df_test)
districts_all, time_blocks_all = get_manual_options(df_test)

# These fixed demo cases are here for presentation stability.
# They let you immediately show one red case and one green case without searching live.
DEMO_CASES = {
    "High-risk demo": {
        "district": 4,
        "event_date": date(2025, 1, 1),
        "time_block": 1,
    },
    "Low-risk demo": {
        "district": 16,
        "event_date": date(2025, 1, 4),
        "time_block": 2,
    },
}


with st.sidebar:
    # Keep the sidebar short and clear. The navigation should be obvious during demo.
    st.markdown("## Team 18")
    st.markdown("### Hotspot Dashboard")
    st.caption("Chicago crime hotspot prediction")

    page = st.radio(
        "Navigate",
        ["Overview", "Single Prediction", "Daily Risk Scan"],
    )


st.title("Chicago Crime Hotspot Prediction Dashboard")
st.caption("Phase 3 deployment app (POC/MVP)")


def render_overview():
    st.header("Overview")

    # A few quick metrics help the audience understand the scale of the demo data.
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test rows", f"{len(df_test):,}")
    c2.metric("Districts", f"{df_test['district'].nunique():,}")
    c3.metric("Time blocks", f"{df_test['time_block'].nunique():,}")
    c4.metric("Test year", "2025")

    st.markdown(
        """
        ### What this app does
        This application predicts whether a **Chicago police district** will become a **crime hotspot in the next 4-hour block**.

        ### Deployment setup
        - **Task**: next-block hotspot prediction
        - **Model**: HistGradientBoosting
        - **Training period**: 2015–2024
        - **Testing / demonstration period**: 2025
        - **Purpose**: proof-of-concept (POC/MVP) deployment for prediction workflow demonstration
        """
    )

    st.subheader("How to use")
    st.markdown(
        """
        1. Use **Single Prediction** to inspect one district/date/time-block record.  
        2. Use **Daily Risk Scan** to compare selected districts on a chosen date.  
        3. For demo, use the built-in **High-risk** and **Low-risk** sample cases.  
        """
    )

    # This mapping matters because block numbers alone are not intuitive.
    st.subheader("Time block mapping")
    mapping_df = pd.DataFrame(
        {
            "Time Block": [f"Block {k}" for k in TIME_BLOCK_LABELS.keys()],
            "Time Window": list(TIME_BLOCK_LABELS.values()),
        }
    )
    st.dataframe(mapping_df, use_container_width=True, hide_index=True)

    st.info(
        "This deployment focuses on prediction functionality and business interpretation. "
        "It is a simple working POC rather than a real-time operational system."
    )

    # I keep the validation notes hidden by default.
    # They are useful for the assignment, but not important for the first impression.
    with st.expander("Built-in validation and error handling"):
        st.markdown(
            """
            - Missing files are checked before loading  
            - Missing required columns trigger clear error messages  
            - Empty filter results show warnings instead of crashing  
            - Prediction errors are caught and displayed to the user  
            - Date selection is limited to the 2025 test/demo period  
            """
        )


def render_single_prediction():
    st.header("Single Prediction")
    st.write(
        "Generate a next-block hotspot prediction for one selected district/date/time-window record."
    )

    # Demo mode is great for presentation.
    # Manual mode is there in case you want to show that the app is interactive.
    mode = st.radio(
        "Selection mode",
        ["Demo Cases", "Manual Selection"],
        horizontal=True,
    )

    if mode == "Demo Cases":
        demo_name = st.selectbox("Choose a demo case", list(DEMO_CASES.keys()))
        chosen = DEMO_CASES[demo_name]

        district = chosen["district"]
        selected_date = chosen["event_date"]
        time_block = chosen["time_block"]

        st.caption(
            f"Loaded case: {format_district(district)} | {selected_date} | {format_time_block(time_block)}"
        )

    else:
        col1, col2, col3 = st.columns(3)

        district_display = [format_district(d) for d in districts_all]
        district_map = dict(zip(district_display, districts_all))

        time_display = [format_time_block(tb) for tb in time_blocks_all]
        time_map = dict(zip(time_display, time_blocks_all))

        selected_district_display = col1.selectbox("District", district_display)

        # Use a date picker instead of a huge dropdown.
        # It feels much more natural when the year contains many dates.
        selected_date = col2.date_input(
            "Date",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
        )

        selected_time_display = col3.selectbox("Time Block", time_display)

        district = district_map[selected_district_display]
        time_block = time_map[selected_time_display]

    if st.button("Predict", type="primary"):
        matched = filter_single_record(df_test, district, selected_date, time_block)

        if matched.empty:
            st.warning("No matching record found for the selected district/date/time block.")
            return

        if len(matched) > 1:
            st.warning("Multiple matching records found. Using the first matching record.")
            matched = matched.head(1)

        # predict_rows() already wraps the deployed model logic.
        # If something unexpected happens, show a readable error instead of breaking the whole app.
        try:
            pred_df, threshold, _ = predict_rows(matched)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        result_row = pd.concat(
            [matched.reset_index(drop=True), pred_df.reset_index(drop=True)],
            axis=1,
        )

        prob = float(result_row.loc[0, "predicted_probability"])
        pred = int(result_row.loc[0, "predicted_label"])

        render_prediction_banner(prob, pred, threshold)

        # Show the main result summary first.
        c1, c2, c3 = st.columns(3)
        c1.metric("District", format_district(result_row.loc[0, "district"]))
        c2.metric("Time Window", TIME_BLOCK_LABELS.get(int(result_row.loc[0, "time_block"]), "Unknown"))
        c3.metric("Predicted Probability", f"{prob:.3f}")

        # The badge is more visual than plain text and works better in demo.
        badge_html = (
            '<span class="risk-badge-high">HIGH RISK</span>'
            if pred == 1 else
            '<span class="risk-badge-low">LOW RISK</span>'
        )
        st.markdown(badge_html, unsafe_allow_html=True)

        st.markdown(
            f"**Official model label:** {hotspot_text(pred)}  \n"
            f"**Decision threshold:** {threshold:.2f}"
        )

        if "target_hotspot_next_block" in result_row.columns:
            actual = int(result_row.loc[0, "target_hotspot_next_block"])
            st.markdown(f"**Actual label:** {hotspot_text(actual)}")

        st.subheader("Selected sample summary")
        st.dataframe(
            build_display_sample_row(result_row.iloc[0]),
            use_container_width=True,
            hide_index=True,
        )


def render_daily_risk_scan():
    st.header("Daily Risk Scan")
    st.write(
        "Scan selected districts on a chosen date and identify the highest-risk district/time-window combinations."
    )

    col1, col2 = st.columns([1, 2])

    # Same date restriction here. Keep everything inside the 2025 demo range.
    selected_date = col1.date_input(
        "Select a date",
        value=min_date,
        min_value=min_date,
        max_value=max_date,
        key="daily_scan_date",
    )

    district_display = [format_district(d) for d in districts_all]
    reverse_district_map = {format_district(d): d for d in districts_all}

    selected_district_display = col2.multiselect(
        "Filter districts (optional)",
        district_display,
        default=district_display[:5] if len(district_display) >= 5 else district_display,
    )

    selected_districts = [reverse_district_map[x] for x in selected_district_display]

    if st.button("Run Daily Risk Scan"):
        batch_df = filter_batch_records(
            df_test,
            selected_date=selected_date,
            selected_districts=selected_districts,
        )

        if batch_df.empty:
            st.warning("No records found for the selected date and district filters.")
            return

        try:
            pred_df, threshold, _ = predict_rows(batch_df)
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")
            return

        result = pd.concat(
            [batch_df.reset_index(drop=True), pred_df.reset_index(drop=True)],
            axis=1,
        )

        # Sort descending because this page is about risk ranking.
        result = result.sort_values("predicted_probability", ascending=False).reset_index(drop=True)
        metrics = compute_daily_metrics(result)

        # These four are the most presentation-friendly summary metrics.
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows scanned", f"{metrics['rows_scanned']:,}")
        c2.metric("Predicted hotspots", f"{metrics['predicted_hotspots']:,}")
        c3.metric("Highest probability", f"{metrics['highest_probability']:.3f}")
        c4.metric("Average probability", f"{metrics['average_probability']:.3f}")

        st.write(f"**Decision threshold used:** {threshold:.2f}")

        # Same-day recall/precision can look unstable and may distract from the main story,
        # so keep them hidden unless someone really wants to inspect them.
        with st.expander("Optional evaluation snapshot for selected date"):
            if metrics["actual_hotspots"] is not None:
                c5, c6, c7 = st.columns(3)
                c5.metric("Actual hotspots", f"{metrics['actual_hotspots']:,}")
                c6.metric("Correct hotspot alerts", f"{metrics['correct_hotspot_alerts']:,}")
                c7.metric("Daily recall", f"{metrics['daily_recall']:.2%}")
                st.caption(
                    "These are same-day evaluation indicators only. "
                    "The main purpose of this page is risk ranking and prioritization."
                )

        risk_table = prepare_risk_scan_table(result)

        st.subheader("Top 10 highest-risk records")
        st.dataframe(risk_table.head(10), use_container_width=True, hide_index=True)

        # Show only a limited number of bars.
        # If too many labels are shown, the chart becomes unreadable during demo.
        plot_df = result.copy().head(12)
        plot_df["district_label"] = plot_df["district"].apply(format_district)
        plot_df["time_label"] = plot_df["time_block"].apply(lambda x: TIME_BLOCK_LABELS.get(x, "Unknown"))
        plot_df["record_label"] = plot_df["district_label"] + " | " + plot_df["time_label"]
        plot_df["risk_flag"] = plot_df["predicted_label"].map({1: "High Risk", 0: "Low Risk"})

        fig = px.bar(
            plot_df,
            x="record_label",
            y="predicted_probability",
            color="risk_flag",
            color_discrete_map={
                "High Risk": "#d62728",
                "Low Risk": "#4e79a7",
            },
            hover_data={
                "district_label": True,
                "time_label": True,
                "predicted_probability": ":.3f",
                "risk_flag": True,
                "record_label": False,
            },
            title="Top predicted risk records",
        )

        fig.update_layout(
            xaxis_title="District and time window",
            yaxis_title="Predicted probability",
            xaxis_tickangle=-25,
            legend_title="Risk level",
            height=540,
            font=dict(size=15),
            title_font=dict(size=22),
        )

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Show full prediction table"):
            st.dataframe(risk_table, use_container_width=True, hide_index=True)


if page == "Overview":
    render_overview()
elif page == "Single Prediction":
    render_single_prediction()
else:
    render_daily_risk_scan()

