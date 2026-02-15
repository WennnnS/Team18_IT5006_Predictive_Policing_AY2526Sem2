import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import HeatMap
from pathlib import Path
import streamlit.components.v1 as components


from src.placeholder import fake_df

#.\.venv\Scripts\Activate.ps1 激活虚拟环境
#python -m streamlit run dashboard\app.py  启动命令


st.set_page_config(page_title="Team18 - Phase1 Dashboard", layout="wide")

df = fake_df()
FIG_DIR = Path("docs/figures")


st.sidebar.title("Team18 Dashboard")
page = st.sidebar.radio("Navigate", ["Overview", "Temporal", "Spatial"])

st.sidebar.header("Filters")
year = st.sidebar.selectbox("Year", sorted(df["year"].unique()))
crime_type = st.sidebar.selectbox("Primary Type", sorted(df["Primary Type"].unique()))

filtered = df[(df["year"] == year) & (df["Primary Type"] == crime_type)]

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

    yearly_png = FIG_DIR / "temporal_yearly.png"
    heatmap_png = FIG_DIR / "weekday_hour_heatmap.png"

    st.subheader("Yearly Trend (exported)")
    if yearly_png.exists():
        st.image(str(yearly_png), use_container_width=True)
    else:
        st.warning(f"Missing: {yearly_png}")
    
    st.subheader("Weekday × Hour Heatmap (exported)")
    if heatmap_png.exists():
        st.image(str(heatmap_png), use_container_width=True)
    else:
        st.warning(f"Missing: {heatmap_png}")

    # fallback placeholder
    if (not yearly_png.exists()) or (not heatmap_png.exists()):
        st.info("Showing placeholder chart because one or more exported figures are missing.")
        hourly = filtered.groupby("hour").size()
        st.bar_chart(hourly)


elif page == "Spatial":
    st.title("Spatial Hotspots")

    hotspot_html = FIG_DIR / "hotspot_map.html"
    st.subheader("Hotspot Map (exported)")

    if hotspot_html.exists():
        html = hotspot_html.read_text(encoding="utf-8", errors="ignore")
        components.html(html, height=650, scrolling=True)
    else:
        st.warning(f"Missing: {hotspot_html}")
        st.info("Showing placeholder heatmap because exported hotspot_map.html is missing.")
        m = folium.Map(location=[41.88, -87.63], zoom_start=10)
        heat = filtered[["Latitude", "Longitude"]].values.tolist()
        HeatMap(heat, radius=10).add_to(m)
        st_folium(m, width=1100, height=600)

