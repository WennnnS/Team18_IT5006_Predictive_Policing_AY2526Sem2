import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import HeatMap
from pathlib import Path
import streamlit.components.v1 as components
import pandas as pd
import numpy as np



def fake_df():
    rng = pd.date_range("2024-01-01", periods=500, freq="H")
    df = pd.DataFrame({
        "Date": rng,
        "Primary Type": np.random.choice(["THEFT", "BATTERY", "BURGLARY"], size=len(rng)),
        "Latitude": 41.88 + np.random.normal(0, 0.03, size=len(rng)),
        "Longitude": -87.63 + np.random.normal(0, 0.03, size=len(rng)),
    })
    df["year"] = df["Date"].dt.year
    df["hour"] = df["Date"].dt.hour
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    df["month"] = df["Date"].dt.month
    df["weekday"] = df["Date"].dt.day_name()

    
    return df


#from src.placeholder import fake_df

#.\.venv\Scripts\Activate.ps1 激活虚拟环境
#python -m streamlit run dashboard\app.py  启动命令


st.set_page_config(page_title="Team18 - Phase1 Dashboard", layout="wide")

#df = fake_df()
DATA_PATH = Path("data/raw/chicago_2015_2024_temporal.csv")

if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
    #st.write("Loaded columns:", list(df.columns))

    # 确保列名一致（CSV 已经是 date / primary Type）
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["hour"] = df["date"].dt.hour
    df["weekday"] = df["date"].dt.day_name()
else:
    st.warning(f"Real data not found at {DATA_PATH}. Using placeholder data.")
    df = fake_df()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.month
    df["weekday"] = df["date"].dt.day_name()

FIG_DIR = Path("docs/figures")


st.sidebar.title("Team18 Dashboard")
#page = st.sidebar.radio("Navigate", ["Overview", "Temporal", "Spatial"])
page = st.sidebar.radio("Navigate", ["Overview", "Temporal", "Spatial", "Correlation"])

st.sidebar.header("Filters")
year = st.sidebar.selectbox("Year", sorted(df["year"].unique()))
show_all_years = st.sidebar.checkbox("Show all years (2015–2024) trend", value=True)
crime_type = st.sidebar.selectbox("Primary Type", sorted(df["primary_type"].unique()))

#filtered = df[(df["year"] == year) & (df["primary_type"] == crime_type)]

if show_all_years:
    base = df[df["primary_type"] == crime_type]   # 不限制 year
else:
    base = df[(df["year"] == year) & (df["primary_type"] == crime_type)]

filtered = base
st.sidebar.caption(f"Filtered rows: {len(filtered):,}")





if page == "Overview":
    st.title("Overview (Skeleton)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total rows", f"{len(df):,}")
    c2.metric("Filtered rows", f"{len(filtered):,}")
    c3.metric("Selected type", crime_type)

    st.subheader("Data preview")
    st.dataframe(filtered.head(20))

elif page == "Temporal":
    st.title("Temporal Patterns")
#交互图
    st.subheader("Trend: Monthly incidents over time")
    # 生成 year-month（月度时间序列）
    tmp = filtered.copy()
    tmp["year_month"] = tmp["date"].dt.to_period("M").dt.to_timestamp()
    monthly_ts = tmp.groupby("year_month").size().sort_index()
    st.caption("Y-axis: incident count per month under selected filters.")
    st.line_chart(monthly_ts)
    st.write("Total incidents in current view:", f"{int(monthly_ts.sum()):,}")
    st.write("Months covered:", len(monthly_ts))


    st.subheader("Interactive: Hourly Distribution")
    hourly = filtered.groupby("hour").size().reindex(range(24), fill_value=0)
    st.caption("Y-axis: number of incidents (count) within selected filters.")
    st.line_chart(hourly)

    st.subheader("Interactive: Monthly Distribution")
    monthly = filtered.groupby("month").size().reindex(range(1, 13), fill_value=0)
    st.caption("Y-axis: number of incidents (count) within selected filters.")
    st.line_chart(monthly)

    st.divider()
    st.subheader("Notebook Exports")
    yearly_png = FIG_DIR / "temporal_yearly.png"
    heatmap_png = FIG_DIR / "weekday_hour_heatmap.png"
#png
    if yearly_png.exists():
        st.image(str(yearly_png), use_container_width=True)
    else:
        st.warning(f"Missing: {yearly_png}")

    if heatmap_png.exists():
        st.image(str(heatmap_png), use_container_width=True)
    else:
        st.warning(f"Missing: {heatmap_png}")



# elif page == "Spatial":
#     st.write(df.columns)
#     st.title("Spatial Hotspots")

#     st.subheader("Spatial Hotspots (Interactive)")
#     crime_type_spatial = st.selectbox(
#         "Crime type (spatial view)",
#         ["All"] + sorted(df["primary_type"].dropna().unique().tolist())
#     )
#     df_spatial = df if crime_type_spatial == "All" else df[df["primary_type"] == crime_type_spatial]
#     st.caption(f"Rows in spatial view: {len(df_spatial):,}")

#     grid_size = st.slider("Grid size (degrees)", 0.005, 0.05, 0.01, step=0.005)


#     st.subheader("Top 10 hotspot grid cells")

#     tmp = df_spatial.dropna(subset=["Latitude", "Longitude"]).copy()
#     tmp["lat_bin"] = (tmp["Latitude"] / grid_size).round().astype(int)
#     tmp["lon_bin"] = (tmp["Longitude"] / grid_size).round().astype(int)

#     hotspots = (
#         tmp.groupby(["lat_bin", "lon_bin"])
#         .size()
#         .sort_values(ascending=False)
#         .head(10)
#         .reset_index(name="count")
#     )

#     # 转回中心点便于理解
#     hotspots["lat_center"] = hotspots["lat_bin"] * grid_size
#     hotspots["lon_center"] = hotspots["lon_bin"] * grid_size

#     st.dataframe(hotspots[["lat_center", "lon_center", "count"]])
#     hotspot_html = FIG_DIR / "hotspot_map.html"
#     st.subheader("Hotspot Map (exported)")

#     if hotspot_html.exists():
#         html = hotspot_html.read_text(encoding="utf-8", errors="ignore")
#         components.html(html, height=650, scrolling=True)
#     else:
#         st.warning(f"Missing: {hotspot_html}")
#         st.info("Showing placeholder heatmap because exported hotspot_map.html is missing.")
#         m = folium.Map(location=[41.88, -87.63], zoom_start=10)
        
#         #heat = filtered[["Latitude", "Longitude"]].values.tolist()
#         heat = tmp[["Latitude", "Longitude"]].values.tolist()
#         HeatMap(heat, radius=10).add_to(m)
#         st_folium(m, width=1100, height=600)
elif page == "Spatial":

    st.title("Spatial Patterns")
    st.caption("Hotspot map generated from 2015–2024 incidents using grid-based aggregation.")

    import streamlit.components.v1 as components
    html_path = Path("docs/figures/hotspot_map.html")

    h = st.slider("Map height", 400, 1000, 650, step=50)

    if not html_path.exists():
        st.error(f"Missing file: {html_path}. Please generate it from the spatial notebook.")
        st.stop()

    html = html_path.read_text(encoding="utf-8", errors="ignore")
    components.html(html, height=h, scrolling=True)

    grid_csv = Path("docs/outputs/grid_counts_2015_2024.csv")
    if grid_csv.exists():
        grid_df = pd.read_csv(grid_csv)
        st.subheader("Top 10 hotspot grid cells")
        top10 = grid_df.sort_values("count", ascending=False).head(10)
        st.dataframe(top10[["lat_center","lon_center","count"]])

    else:
        st.info("Grid summary CSV not found (optional).")

    

    # st.caption("Hotspots are identified via grid-based aggregation (GRID=0.01°) using 2015–2024 data.")

    # # 1) Show hotspot map (HTML)
    # html_path = Path("docs/figures/hotspot_map.html")
    # if html_path.exists():
    #     import streamlit.components.v1 as components
    #     html = html_path.read_text(encoding="utf-8")
    #     h = st.slider("Map height", 400, 1000, 650, step=50)
    #     components.html(html, height=h, scrolling=True)
    #     st.success(f"Loaded: {html_path}")
    # else:
    #     st.error(f"Missing file: {html_path}. Please run the spatial notebook to generate it.")

    # st.divider()

    # # 2) Show Top hotspots table if grid csv exists
    # grid_csv = Path("docs/outputs/grid_counts_2015_2024.csv")  # 你改成你实际生成的文件名
    # if grid_csv.exists():
    #     grid_df = pd.read_csv(grid_csv)
    #     st.subheader("Top 10 hotspot grid cells")
    #     # 兼容列名
    #     for c in ["count", "crime_count", "n"]:
    #         if c in grid_df.columns:
    #             count_col = c
    #             break
    #     else:
    #         count_col = "count"

    #     show_cols = [c for c in ["lat_center", "lon_center", count_col] if c in grid_df.columns]
    #     top10 = grid_df.sort_values(count_col, ascending=False).head(10)
    #     st.dataframe(top10[show_cols])
    #     st.success(f"Loaded: {grid_csv}")
    # else:
    #     st.warning(f"Grid CSV not found at {grid_csv}. If you have it elsewhere, update this path.")




elif page == "Correlation":
    st.title("Crime Correlation Analysis")

    st.subheader("Crime Type × Month Heatmap")

    

    # 使用全数据，不按 year 限制（更有意义）
    # pivot = (
    #     df.groupby(["primary_type", "month"])
    #       .size()
    #       .unstack(fill_value=0)
    #       .sort_index()
    # )

    # # --- controls ---
    # topN = st.slider("Top N crime types (by total incidents)", 5, 40, 20)
    # min_total = st.slider("Minimum total incidents per type (to avoid tiny-sample bias)", 50, 2000, 300, step=50)

    # # total counts per type
    # type_totals = df["primary_type"].value_counts()

    # # choose topN, then drop tiny-sample types
    # top_types = type_totals.head(topN).index
    # keep_types = [t for t in top_types if type_totals[t] >= min_total]

    # pivot = pivot.loc[keep_types]
    # pivot_ratio = pivot.div(pivot.mean(axis=1), axis=0)
    pivot_all = (
        df.groupby(["primary_type", "month"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    pivot_ratio_all = pivot_all.div(pivot_all.mean(axis=1), axis=0)


    # #st.caption("Each cell shows the number of incidents for a crime type in a given month (2015–2024).")
    # st.caption("Heatmap shows month-to-month deviation relative to each crime type's average month (ratio > 1 means higher-than-usual).")

    # st.dataframe(pivot)

    # st.write("Heatmap visualization:")
    import matplotlib.pyplot as plt
    import seaborn as sns

    # #fig, ax = plt.subplots(figsize=(10, 6))
    # fig, ax = plt.subplots(figsize=(10, 0.35 * len(pivot_ratio) + 2))
    # import numpy as np
    # upper = np.percentile(pivot_ratio.values, 95)

    # sns.heatmap(
    #     pivot_ratio,
    #     cmap="YlGnBu",
    #     ax=ax,
    #     # vmin=0,
    #     # vmax=upper

    #     center=1,
    #     vmin=0.7,
    #     vmax=upper
    # )

    

    # ax.set_xlabel("Month")
    # ax.set_ylabel("Crime Type")
    # st.pyplot(fig)
    show_overview = st.checkbox("Show overview heatmap (all types)", value=False)

    if show_overview:
        st.subheader("Overview: All crime types (ratio to type-average)")

        st.write("Overview table (counts):")
        st.dataframe(pivot_all)

        import numpy as np
        upper_all = np.percentile(pivot_ratio_all.values, 95)

        #fig, ax = plt.subplots(figsize=(10, 0.25 * len(pivot_ratio_all) + 2))
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(
            pivot_ratio_all,
            cmap="YlGnBu",
            ax=ax,
            center=1,
            vmin=0.7,
            vmax=upper_all
        )
        ax.set_xlabel("Month")
        ax.set_ylabel("Crime Type")
        ax.tick_params(axis="y", labelsize=7)
        ax.set_yticks(range(len(pivot_ratio_all.index)))
        ax.set_yticklabels(pivot_ratio_all.index, rotation=0)
        st.pyplot(fig)

    else:
        st.subheader("Focused view: Top N crime types (filter tiny-sample bias)")

        topN = st.slider("Top N crime types (by total incidents)", 5, 40, 20)
        min_total = st.slider("Minimum total incidents per type", 50, 2000, 300, step=50)

        type_totals = df["primary_type"].value_counts()
        top_types = type_totals.head(topN).index
        keep_types = [t for t in top_types if type_totals[t] >= min_total]

        pivot_focus = pivot_all.loc[keep_types]
        pivot_ratio_focus = pivot_focus.div(pivot_focus.mean(axis=1), axis=0)

        upper_focus = np.percentile(pivot_ratio_focus.values, 95)

        #fig2, ax2 = plt.subplots(figsize=(10, 0.35 * len(pivot_ratio_focus) + 2))
        fig2, ax2 = plt.subplots(figsize=(9, 6))
        sns.heatmap(
            pivot_ratio_focus,
            cmap="YlGnBu",
            ax=ax2,
            center=1,
            vmin=0.7,
            vmax=upper_focus
        )
        ax2.set_xlabel("Month")
        ax2.set_ylabel("Crime Type")
        ax2.tick_params(axis="y", labelsize=8)
        ax2.tick_params(axis="x", labelsize=8)

        st.pyplot(fig2)
