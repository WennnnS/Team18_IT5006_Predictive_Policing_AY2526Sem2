from pathlib import Path
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "phase2"
MODEL_DIR = PROJECT_ROOT / "notebooks" / "phase2" / "models"

TEST_FEATURED_PATH = DATA_DIR / "test_panel_featured.csv"
TRAIN_FEATURED_PATH = DATA_DIR / "train_panel_featured.csv"
MODEL_PATH = MODEL_DIR / "hist_gradient_boosting.joblib"

DATE_COLUMN = "event_date"
REQUIRED_COLUMNS = ["district", "event_date", "time_block"]


def check_file_exists(path: Path, label: str) -> None:
    if not path.exists():
        st.error(f"{label} not found: {path}")
        st.stop()


@st.cache_data
def load_test_data() -> pd.DataFrame:
    check_file_exists(TEST_FEATURED_PATH, "Test dataset")
    df = pd.read_csv(TEST_FEATURED_PATH)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"Missing required columns in test dataset: {missing}")
        st.stop()

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    df = df.dropna(subset=[DATE_COLUMN]).copy()
    return df


@st.cache_data
def load_train_data() -> pd.DataFrame:
    if not TRAIN_FEATURED_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(TRAIN_FEATURED_PATH)
    if DATE_COLUMN in df.columns:
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    return df


def get_available_options(df: pd.DataFrame):
    districts = sorted(df["district"].dropna().unique().tolist())
    dates = sorted(df[DATE_COLUMN].dt.date.dropna().unique().tolist())
    time_blocks = sorted(df["time_block"].dropna().unique().tolist())
    return districts, dates, time_blocks


def filter_single_record(
    df: pd.DataFrame,
    district,
    selected_date,
    time_block,
) -> pd.DataFrame:
    matched = df[
        (df["district"] == district)
        & (df[DATE_COLUMN].dt.date == selected_date)
        & (df["time_block"] == time_block)
    ].copy()
    return matched


def filter_batch_records(
    df: pd.DataFrame,
    selected_date=None,
    selected_districts=None,
):
    out = df.copy()

    if selected_date is not None:
        out = out[out[DATE_COLUMN].dt.date == selected_date]

    if selected_districts:
        out = out[out["district"].isin(selected_districts)]

    return out