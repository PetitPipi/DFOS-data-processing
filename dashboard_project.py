import streamlit as st
import polars as pl
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import glob
import pandas as pd

# --- CONFIGURATION ---
st.set_page_config(page_title="ODiSI Explorer v17", layout="wide")
st.title("📊 Project Data Explorer (Smart Metadata)")

# --- 1. FILE SELECTION ---
files = sorted(glob.glob("*.parquet"))
if not files:
    st.error("No .parquet files found.")
    st.stop()

st.sidebar.header("📂 Select Data")
selected_file = st.sidebar.selectbox("File:", files)

# --- 2. LOAD METADATA & CONFIG ---
json_file = selected_file.replace(".parquet", "_metadata.json")
sensor_locs = None
gage_names = None
data_type = "Unknown"
unit = "-"
metadata_status = "❌ Missing headers"

if os.path.exists(json_file):
    with open(json_file, "r") as f:
        try:
            meta = json.load(f)
            
            # Load Sensors
            if isinstance(meta, dict):
                sensor_locs = np.array(meta.get("locations", []), dtype=float)
                gage_names = meta.get("gages", [])
                # LOAD TYPE FROM HEADER (New!)
                data_type = meta.get("sensor_type", "Strain")
                unit = meta.get("units", "με")
            
            if sensor_locs is not None and len(sensor_locs) > 0:
                metadata_status = f"✅ Loaded {len(sensor_locs)} sensors"
        except:
            metadata_status = "⚠️ Corrupt JSON"

# Configure Plot based on Type
if "temp" in data_type.lower():
    color_scale = "Inferno"
    default_min, default_max = 20, 60
else:
    # Default to Strain settings
    color_scale = "RdBu_r"
    default_min, default_max = -50, 50

# Sidebar Info
st.sidebar.info(f"**Type:** {data_type}\n**Unit:** {unit}")
st.sidebar.text(metadata_status)

# --- 3. METADATA TIME LOAD ---
lf = pl.scan_parquet(selected_file)
try:
    # Filter "Tare" rows to get real time
    clean_lf = lf.filter(
        (pl.col("Time").str.len_bytes() > 10) & 
        (pl.col("Time") != "Tare")
    )
    time_info = clean_lf.select([
        pl.min("Time").alias("start"),
        pl.max("Time").alias("end"),
        pl.len().alias("count")
    ]).collect()
    
    start_ts = time_info["start"][0]
    end_ts = time_info["end"][0]
    total_rows = time_info["count"][0]
    
    st.sidebar.caption(f"📅 Start: {start_ts}")
    st.sidebar.caption(f"📅 End:   {end_ts}")
    st.sidebar.caption(f"🔢 Scans: {total_rows}")
except:
    st.error("Could not read timestamps.")
    st.stop()

# --- 4. SIDEBAR TOOLS ---
st.sidebar.markdown("---")
st.sidebar.header("🛠️ Tools")

# Spatial Zoom
with st.sidebar.expander("🔎 Spatial Crop (Zoom)"):
    min_m_def = float(sensor_locs[0]) if sensor_locs is not None else 0.0
    max_m_def = float(sensor_locs[-1]) if sensor_locs is not None else 50.0
    zoom_min = st.number_input("Min (m)", value=min_m_def, step=0.01)
    zoom_max = st.number_input("Max (m)", value=max_m_def, step=0.01)

    if sensor_locs is not None:
        idx_start = (np.abs(sensor_locs - zoom_min)).argmin()
        idx_end = (np.abs(sensor_locs - zoom_max)).argmin()
        if idx_start > idx_end: idx_start, idx_end = idx_end, idx_start
        if idx_start == idx_end: idx_end += 1 
    else:
        idx_start = int(zoom_min / 0.0013)
        idx_end = int(zoom_max / 0.0013)

# Calculator
with st.sidebar.expander("🧮 Index Calculator"):
    calc_mode = st.radio("Convert:", ["Index ➡ Meters", "Meters ➡ Index"])
    if sensor_locs is not None:
        if calc_mode == "Index ➡ Meters":
            i_val = st.number_input("Index:", 0, len(sensor_locs)-1, 0)
            res_m = sensor_locs[i_val]
            res_name = gage_names[i_val] if gage_names and i_val < len(gage_names) else ""
            st.write(f"= **{res_m:.4f} m**")
            if res_name: st.caption(f"({res_name})")
        else:
            m_val = st.number_input("Meters:", min_m_def, max_m_def, min_m_def)
            found = (np.abs(sensor_locs - m_val)).argmin()
            st.write(f"≈ **Index {found}**")
    else:
        st.warning("No metadata.")

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["🔥 Heatmap", "📏 Spatial Profile", "📈 Time History", "✅ Verify"])

# Helper function
def get_data_slice(t1, t2):
    return (
        pl.scan_parquet(selected_file)
        .filter((pl.col("Time") >= t1) & (pl.col("Time") <= t2))
        .collect()
    )

# ==========================================
# TAB 1: HEATMAP
# ==========================================
with tab1:
    st.write(f"### ⏱️ {data_type} Heatmap")
    
    c1, c2 = st.columns(2)
    t1 = c1.text_input("Start Time", value=start_ts, key="t1_hm")
    t2 = c2.text_input("End Time", value=end_ts, key="t2_hm")
    t2_q = t2 + ".999999" if len(t2) == 19 else t2

    sc1, sc2, sc3 = st.columns(3)
    orientation = sc1.selectbox("Axes", ["Standard (X=m)", "Rotated (X=t)"])
    use_manual = sc2.checkbox("Manual Range", value=False, key="hm_manual")
    z_min, z_max = None, None
    if use_manual:
        c_min, c_max = st.columns(2)
        z_min = c_min.number_input("Min", value=default_min, key="hm_min")
        z_max = c_max.number_input("Max", value=default_max, key="hm_max")

    if st.button("Render Heatmap", type="primary"):
        with st.spinner("Rendering..."):
            slice_df = get_data_slice(t1, t2_q)
            if len(slice_df) == 0: 
                st.warning("No data found.")
                st.stop()

            full_matrix = np.vstack(slice_df["Data_Array"].to_numpy())
            safe_end = min(full_matrix.shape[1], idx_end + 1)
            matrix = full_matrix[:, idx_start : safe_end]
            time_labels = slice_df["Time"].to_list()
            
            if sensor_locs is not None:
                if full_matrix.shape[1] > len(sensor_locs):
                     pad = np.arange(1, full_matrix.shape[1] - len(sensor_locs) + 1) * 0.0013 + sensor_locs[-1]
                     all_locs = np.concatenate([sensor_locs, pad])
                else:
                     all_locs = sensor_locs
                current_locs = all_locs[idx_start : safe_end]
            else:
                current_locs = np.arange(idx_start, safe_end) * 0.0013

            if orientation == "Standard (X=m)":
                plot_m = matrix
                x, y = current_locs, time_labels
                labels = dict(x="Location (m)", y="Time", color=unit)
                if plot_m.shape[0] > 1500: # Downsample Time
                    skip = plot_m.shape[0] // 1500
                    plot_m, y = plot_m[::skip, :], y[::skip]
            else:
                plot_m = matrix.T
                x, y = time_labels, current_locs
                labels = dict(x="Time", y="Location (m)", color=unit)
                if plot_m.shape[0] > 1500: # Downsample Meters
                    skip = plot_m.shape[0] // 1500
                    plot_m, x = plot_m[::skip, :], x[::skip]

            fig = px.imshow(
                plot_m, aspect='auto', x=x, y=y, labels=labels, 
                color_continuous_scale=color_scale, 
                origin='lower', zmin=z_min, zmax=z_max
            )
            st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 2: SPATIAL PROFILE
# ==========================================
with tab2:
    st.write(f"### 📏 {data_type} Profile (Single Timestep)")
    
    row_idx = st.slider("Select Time Index:", 0, total_rows-1, 0)
    compare_str = st.text_input("Overlay Time Indices (comma sep):", help="e.g. 0, 500, 1000")
    
    indices_to_plot = [row_idx]
    if compare_str.strip():
        try:
            indices_to_plot.extend([int(x.strip()) for x in compare_str.split(",") if x.strip()])
        except: pass

    if st.button("Plot Profile"):
        with st.spinner("Extracting..."):
            q = (
                clean_lf
                .with_row_index("idx")
                .filter(pl.col("idx").is_in(indices_to_plot))
                .collect()
            )
            
            if len(q) > 0:
                fig = go.Figure()
                for row in q.iter_rows(named=True):
                    arr = np.array(row["Data_Array"])
                    safe_end = min(len(arr), idx_end + 1)
                    y_vals = arr[idx_start : safe_end]
                    
                    if sensor_locs is not None:
                        if len(sensor_locs) >= len(arr):
                             curr = sensor_locs[:len(arr)]
                        else:
                             pad = np.arange(1, len(arr) - len(sensor_locs) + 1) * 0.0013 + sensor_locs[-1]
                             curr = np.concatenate([sensor_locs, pad])
                        x_vals = curr[idx_start : safe_end]
                    else:
                        x_vals = np.arange(idx_start, safe_end) * 0.0013

                    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name=f"Idx {row['idx']} ({row['Time']})"))
                
                fig.update_layout(xaxis_title="Location (m)", yaxis_title=f"{data_type} ({unit})")
                st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 3: TIME HISTORY
# ==========================================
with tab3:
    st.write(f"### 📈 {data_type} History (Multi-Gage)")
    c1, c2 = st.columns(2)
    t1_g = c1.text_input("Start Time (Gage)", value=start_ts, key="gh_t1")
    t2_g = c2.text_input("End Time (Gage)", value=end_ts, key="gh_t2")
    
    default_locs = f"{(min_m_def+max_m_def)/2:.2f}"
    loc_input = st.text_input("Locations (m, comma-sep):", value=default_locs)

    if st.button("Plot History"):
        t2_qg = t2_g + ".999999" if len(t2_g) == 19 else t2_g
        indices, names = [], []
        if loc_input.strip():
            try:
                targets = [float(x.strip()) for x in loc_input.split(",") if x.strip()]
                for t in targets:
                    if sensor_locs is not None:
                        idx = (np.abs(sensor_locs - t)).argmin()
                        real_val = sensor_locs[idx]
                        name_str = f"{real_val:.4f}m"
                        
                        # Add Gage Name if available
                        if gage_names and idx < len(gage_names):
                            g_name = gage_names[idx]
                            if "Loc_" not in g_name: name_str += f" ({g_name})"
                        
                        names.append(name_str)
                    else:
                        idx = int(t / 0.0013)
                        names.append(f"Idx {idx}")
                    indices.append(idx)
            except: pass

        if indices:
            cols = [pl.col("Time")]
            for idx, name in zip(indices, names):
                cols.append(pl.col("Data_Array").list.get(int(idx)).alias(name))
            
            df_hist = pl.scan_parquet(selected_file).filter((pl.col("Time") >= t1_g) & (pl.col("Time") <= t2_qg)).select(cols).collect()
            fig = px.line(df_hist, x="Time", y=names, title=f"{data_type} History")
            fig.update_layout(yaxis_title=f"{data_type} ({unit})", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 4: VERIFY
# ==========================================
with tab4:
    if sensor_locs is not None:
        v_df = pd.DataFrame({"Idx": range(len(sensor_locs)), "Loc (m)": sensor_locs})
        if gage_names:
            v_df["Gage Name"] = gage_names[:len(sensor_locs)]
        st.dataframe(v_df, width=800)
    else:
        st.warning("No metadata.")