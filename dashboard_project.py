import streamlit as st
import polars as pl
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import glob
import pandas as pd
import tempfile
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def _build_spatial_axis(sensor_locs_arr, n_points):
    if sensor_locs_arr is not None:
        if n_points > len(sensor_locs_arr):
            pad = np.arange(1, n_points - len(sensor_locs_arr) + 1) * 0.0026 + sensor_locs_arr[-1]
            return np.concatenate([sensor_locs_arr, pad])
        return sensor_locs_arr[:n_points]
    return np.arange(n_points) * 0.0026


def create_strain_animation_gif(
    strain_matrix,
    x_values,
    time_labels,
    reverse_x=False,
    fps=8,
    y_min=None,
    y_max=None
):
    if reverse_x:
        x_plot = x_values[::-1]
        z_plot = strain_matrix[:, ::-1]
    else:
        x_plot = x_values
        z_plot = strain_matrix

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=120)
    line, = ax.plot(x_plot, z_plot[0], color="#1769aa", linewidth=2)
    title_txt = ax.text(
        0.01, 0.98, "", transform=ax.transAxes, va="top", ha="left",
        fontsize=10, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
    )

    ax.set_xlabel("Location (m)")
    ax.set_ylabel("Strain (microstrain)")
    ax.grid(alpha=0.25)
    ax.set_xlim(x_plot[0], x_plot[-1])

    if y_min is None:
        y_min = float(np.nanmin(z_plot))
    if y_max is None:
        y_max = float(np.nanmax(z_plot))
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    pad = 0.05 * max(1.0, y_max - y_min)
    ax.set_ylim(y_min - pad, y_max + pad)

    def _update(frame_idx):
        line.set_ydata(z_plot[frame_idx])
        title_txt.set_text(f"Time: {time_labels[frame_idx]}")
        return line, title_txt

    ani = FuncAnimation(fig, _update, frames=len(time_labels), interval=max(1, int(1000 / fps)), blit=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        ani.save(tmp_path, writer=PillowWriter(fps=fps))
        with open(tmp_path, "rb") as f:
            gif_bytes = f.read()
        return gif_bytes
    finally:
        plt.close(fig)
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

# --- CONFIGURATION ---
st.set_page_config(page_title="ODiSI Explorer v18", layout="wide")
st.title("🚀 ODiSI Explorer v18 (Custom Defaults)")

# --- 1. FILE SELECTION ---
files = sorted(glob.glob("*.parquet"))
if not files:
    st.error("No .parquet files found! Run the converter first.")
    st.stop()

st.sidebar.header("📂 Data Source")
selected_file = st.sidebar.selectbox("Select File:", files)

# --- 2. LOAD METADATA & TIME INDEX ---
try:
    lf = pl.scan_parquet(selected_file)
    
    # Collect just the Time column for calculations
    time_df = lf.select("Time").collect()
    all_times = time_df["Time"]
    
    start_ts = all_times[0]
    end_ts = all_times[-1]
    total_rows = len(all_times)
    
    st.sidebar.caption(f"📅 Start: {start_ts}")
    st.sidebar.caption(f"📅 End:   {end_ts}")
    st.sidebar.caption(f"🔢 Scans: {total_rows}")
    
except Exception as e:
    st.error(f"Error loading file: {e}")
    st.stop()

# Load Sensor Locations
json_file = selected_file.replace(".parquet", "_headers.json")
sensor_locs = None
metadata_status = "❌ Missing headers"

if os.path.exists(json_file):
    with open(json_file, "r") as f:
        try:
            loaded_locs = json.load(f)
            if isinstance(loaded_locs, list):
                sensor_locs = np.array(loaded_locs, dtype=float)
                metadata_status = f"✅ Loaded {len(sensor_locs)} sensors"
        except:
            metadata_status = "⚠️ Corrupt JSON"

st.sidebar.text(metadata_status)

# --- 3. SIDEBAR TOOLS ---
st.sidebar.markdown("---")
st.sidebar.header("🛠️ Tools")

# A. Spatial Zoom
with st.sidebar.expander("🔎 Spatial Crop (Zoom)"):
    min_m_def = float(sensor_locs[0]) if sensor_locs is not None else 0.0
    max_m_def = float(sensor_locs[-1]) if sensor_locs is not None else 50.0
    zoom_min = st.number_input("Min (m)", value=min_m_def, step=0.1)
    zoom_max = st.number_input("Max (m)", value=max_m_def, step=0.1)

    if sensor_locs is not None:
        idx_start = (np.abs(sensor_locs - zoom_min)).argmin()
        idx_end = (np.abs(sensor_locs - zoom_max)).argmin()
        if idx_start > idx_end: idx_start, idx_end = idx_end, idx_start
        if idx_start == idx_end: idx_end += 1 
    else:
        idx_start = int(zoom_min / 0.0026)
        idx_end = int(zoom_max / 0.0026)

# B. Index Calculator (Space)
with st.sidebar.expander("📏 Spatial Index Calculator"):
    calc_mode = st.radio("Convert:", ["Index ➡ Meters", "Meters ➡ Index"], key="space_calc")
    if sensor_locs is not None:
        if calc_mode == "Index ➡ Meters":
            i_val = st.number_input("Index:", 0, len(sensor_locs)-1, 0)
            st.write(f"= **{sensor_locs[i_val]:.4f} m**")
        else:
            m_val = st.number_input("Meters:", min_m_def, max_m_def, min_m_def)
            found = (np.abs(sensor_locs - m_val)).argmin()
            st.write(f"≈ **Index {found}**")
    else:
        st.warning("No metadata.")

# C. Time Calculator
with st.sidebar.expander("⏰ Time Index Calculator"):
    time_mode = st.radio("Convert:", ["Index ➡ Time", "Time ➡ Index"], key="time_calc")
    
    if time_mode == "Index ➡ Time":
        row_input = st.number_input("Time Index (Row):", 0, total_rows-1, 0)
        found_time = all_times[row_input]
        st.write(f"= **{found_time}**")
    else:
        t_input = st.text_input("Timestamp:", value=str(start_ts))
        try:
            matches = all_times.filter(all_times >= t_input)
            if len(matches) > 0:
                found_idx = (all_times >= t_input).arg_true().first()
                st.write(f"≈ **Index {found_idx}**")
            else:
                st.warning("Time is out of range.")
        except:
            st.error("Invalid format.")

# --- TABS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🔥 Heatmap", "📏 Spatial Profile", "📈 Time History", "🧊 3D Diff Map", "🎬 Animation", "✅ Verify"])

# ==========================================
# TAB 1: HEATMAP
# ==========================================
with tab1:
    st.write(f"### ⏱️ Spatial Heatmap")
    
    c1, c2 = st.columns(2)
    t1 = c1.text_input("Start Time", value=str(start_ts), key="hm_start")
    t2 = c2.text_input("End Time", value=str(end_ts), key="hm_end")
    t2_q = t2 + ".999999" if len(t2) == 19 else t2

    st.write("#### 🎨 Plot Settings")
    sc1, sc2, sc3, sc4 = st.columns([1, 1, 1, 1])
    
    orientation = sc1.selectbox("Axes", ["Standard (X=m)", "Rotated (X=t)"])
    enable_downsampling = sc2.checkbox("Downsample (Fast)", value=True)
    
    use_manual_scale = sc3.checkbox("Manual Legend Range", value=False)
    z_min, z_max = None, None
    if use_manual_scale:
        z_min = sc4.number_input("Min Strain", value=-50)
        z_max = sc4.number_input("Max Strain", value=50)
    else:
        sc4.info("Auto-scaling active")

    if st.button("🚀 Render Heatmap", type="primary"):
        with st.spinner("Rendering..."):
            slice_df = (
                pl.scan_parquet(selected_file)
                .filter((pl.col("Time") >= t1) & (pl.col("Time") <= t2_q))
                .collect()
            )
            
            if len(slice_df) == 0:
                st.warning("No data found.")
            else:
                full_matrix = np.vstack(slice_df["Data_Array"].to_numpy())
                
                safe_end = min(full_matrix.shape[1], idx_end + 1)
                matrix = full_matrix[:, idx_start : safe_end]
                time_labels = slice_df["Time"].to_list()
                
                if sensor_locs is not None:
                    if full_matrix.shape[1] > len(sensor_locs):
                         pad = np.arange(1, full_matrix.shape[1] - len(sensor_locs) + 1) * 0.0026 + sensor_locs[-1]
                         all_locs = np.concatenate([sensor_locs, pad])
                    else:
                         all_locs = sensor_locs
                    current_locs = all_locs[idx_start : safe_end]
                else:
                    current_locs = np.arange(idx_start, safe_end) * 0.0026

                skip = 1
                if enable_downsampling and matrix.shape[0] > 1500:
                    skip = matrix.shape[0] // 1500
                    plot_matrix = matrix[::skip, :]
                    plot_times = time_labels[::skip]
                else:
                    plot_matrix = matrix
                    plot_times = time_labels

                if orientation == "Standard (X=m)":
                    fig = px.imshow(
                        plot_matrix, aspect='auto', x=current_locs, y=plot_times,
                        labels=dict(x="Location (m)", y="Time", color="Strain"),
                        color_continuous_scale='RdBu_r', origin='lower', zmin=z_min, zmax=z_max
                    )
                else:
                    fig = px.imshow(
                        plot_matrix.T, aspect='auto', x=plot_times, y=current_locs,
                        labels=dict(x="Time", y="Location (m)", color="Strain"),
                        color_continuous_scale='RdBu_r', origin='lower', zmin=z_min, zmax=z_max
                    )
                
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("💾 Export Heatmap Data"):
                    st.write("Save the currently visible data slice to CSV.")
                    export_cols = [f"{l:.4f}m" for l in current_locs]
                    df_export = pl.DataFrame(plot_matrix)
                    df_export.columns = export_cols
                    df_export = df_export.with_columns(pl.Series("Time", plot_times)).select(["Time"] + export_cols)

                    csv_data = df_export.write_csv()
                    st.download_button(
                        label="📥 Download Subset CSV",
                        data=csv_data,
                        file_name=f"ODiSI_Heatmap_{t1}_{t2}.csv",
                        mime="text/csv"
                    )

# ==========================================
# TAB 2: SPATIAL PROFILE
# ==========================================
with tab2:
    st.write("### 📏 Strain Profile (Strain vs. Distance)")
    row_idx = st.slider("Select Time Index:", 0, total_rows-1, 0)
    st.info(f"Selected Time: **{all_times[row_idx]}**")
    
    compare_str = st.text_input("Overlay Time Indices (comma sep):", help="e.g. 0, 500, 1000")
    indices = [row_idx]
    if compare_str.strip():
        try:
            indices.extend([int(x.strip()) for x in compare_str.split(",") if x.strip()])
        except:
            st.error("Invalid input.")

    if st.button("Plot Profiles"):
        with st.spinner("Extracting..."):
            q = (
                pl.scan_parquet(selected_file)
                .with_row_index("idx")
                .filter(pl.col("idx").is_in(indices))
                .collect()
            )
            
            if len(q) > 0:
                fig = go.Figure()
                export_dict = {} 
                x_vals_export = None
                
                for row in q.iter_rows(named=True):
                    arr = np.array(row["Data_Array"])
                    safe_end = min(len(arr), idx_end + 1)
                    y_vals = arr[idx_start : safe_end]
                    
                    if sensor_locs is not None:
                        if len(sensor_locs) >= len(arr):
                             curr = sensor_locs[:len(arr)]
                        else:
                             pad = np.arange(1, len(arr) - len(sensor_locs) + 1) * 0.0026 + sensor_locs[-1]
                             curr = np.concatenate([sensor_locs, pad])
                        x_vals = curr[idx_start : safe_end]
                    else:
                        x_vals = np.arange(idx_start, safe_end) * 0.0026
                    
                    x_vals_export = x_vals

                    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name=f"Idx {row['idx']} ({row['Time']})"))
                    clean_time = row['Time'].replace(":", "-")
                    export_dict[f"Strain_{clean_time}"] = y_vals

                fig.update_layout(xaxis_title="Location (m)", yaxis_title="Strain (με)")
                st.plotly_chart(fig, use_container_width=True)

                if x_vals_export is not None:
                    try:
                        full_export = {"Location_m": x_vals_export, **export_dict}
                        df_profile_export = pl.DataFrame(full_export)
                        csv_profile = df_profile_export.write_csv()
                        st.download_button(
                            label="📥 Download Profiles CSV",
                            data=csv_profile,
                            file_name="ODiSI_Selected_Profiles.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"Could not prepare export: {e}")

# ==========================================
# TAB 3: TIME HISTORY
# ==========================================
with tab3:
    st.write("### 📈 Time History (Multi-Gage)")
    c1, c2 = st.columns(2)
    t1_g = c1.text_input("Start Time (Gage)", value=str(start_ts), key="gh_start")
    t2_g = c2.text_input("End Time (Gage)", value=str(end_ts), key="gh_end")
    t2_qg = t2_g + ".999999" if len(t2_g) == 19 else t2_g

    default_locs = f"{(min_m_def+max_m_def)/2:.2f}"
    loc_input = st.text_input("Locations (m, comma-sep):", value=default_locs)

    if st.button("Plot History"):
        indices, names = [], []
        if loc_input.strip():
            try:
                targets = [float(x.strip()) for x in loc_input.split(",") if x.strip()]
                for t in targets:
                    if sensor_locs is not None:
                        idx = (np.abs(sensor_locs - t)).argmin()
                        names.append(f"{sensor_locs[idx]:.4f}m")
                    else:
                        idx = int(t / 0.0026)
                        names.append(f"Idx {idx}")
                    indices.append(idx)
            except:
                st.error("Invalid input.")

        if indices:
            cols = [pl.col("Time")]
            for idx, name in zip(indices, names):
                cols.append(pl.col("Data_Array").list.get(int(idx)).alias(name))
            
            df_hist = (
                pl.scan_parquet(selected_file)
                .filter((pl.col("Time") >= t1_g) & (pl.col("Time") <= t2_qg))
                .select(cols)
                .collect()
            )
            
            fig = px.line(df_hist, x="Time", y=names, title="Strain History")
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            csv_hist = df_hist.write_csv()
            st.download_button(
                label="📥 Download Time History CSV",
                data=csv_hist,
                file_name="ODiSI_Time_History.csv",
                mime="text/csv"
            )

# ==========================================
# TAB 4: 3D DIFF MAP
# ==========================================
with tab4:
    st.write("### 🧊 3D Strain Difference Map")
    st.markdown("""
    **Map Construction:**
    1. **Calculate:** `Profile(A) - Profile(B)`
    2. **Read:** 5 points (`a, b, c, d, e`) with alternating read ranges.
    3. **Normalize:** `b` and `d` segments are reversed so all Y-axes represent `[-x, +y]`.
    4. **Scale:** Plots are locked to real-world dimensions (1m X = 1m Y).
    """)

    # 1. Select Time Profiles
    c_t1, c_t2 = st.columns(2)
    # UPDATED DEFAULTS: 27405 & 27368
    def_idx_a = 27405 if total_rows > 27405 else total_rows - 1
    def_idx_b = 27368 if total_rows > 27368 else 0
    
    idx_a = c_t1.number_input("Time Index A (Current)", 0, total_rows-1, def_idx_a)
    idx_b = c_t2.number_input("Time Index B (Reference)", 0, total_rows-1, def_idx_b)
    
    c_t1.caption(f"Time: {all_times[idx_a]}")
    c_t2.caption(f"Time: {all_times[idx_b]}")

    st.markdown("---")

    # 2. Configure Points
    st.write("#### 📍 Line Locations (m)")
    c_pts = st.columns(5)
    
    # UPDATED DEFAULTS: 2.223, 8.575, 9.75, 15.95, 17.04
    loc_a = c_pts[0].number_input("Line a (X=2.0)", value=2.223, format="%.4f")
    loc_b = c_pts[1].number_input("Line b (X=1.5)", value=8.575, format="%.4f")
    loc_c = c_pts[2].number_input("Line c (X=1.0)", value=9.750, format="%.4f")
    loc_d = c_pts[3].number_input("Line d (X=0.5)", value=15.950, format="%.4f")
    loc_e = c_pts[4].number_input("Line e (X=0.0)", value=17.040, format="%.4f")

    # 3. Configure Offsets
    st.write("#### ↔️ Read Range Offsets")
    c_off1, c_off2 = st.columns(2)
    # UPDATED DEFAULTS: X=0.3, Y=2.7
    val_x = c_off1.number_input("Offset X (m) [Left bound]", value=0.3, step=0.01)
    val_y = c_off2.number_input("Offset Y (m) [Right bound]", value=2.7, step=0.01)

    # 4. View Mode
    st.markdown("#### 👁️ View Settings")
    view_c1, view_c2, view_c3 = st.columns(3)
    show_3d = view_c1.checkbox("Show 3D Map", value=True)
    show_2d = view_c2.checkbox("Show 2D Heatmap (Interpolated)", value=True)
    
    # Legend controls for the 2D Heatmap
    use_manual_2d = view_c3.checkbox("Manual Legend (2D Only)", value=False)
    zmin_2d, zmax_2d = None, None
    if use_manual_2d:
        c_leg1, c_leg2 = st.columns(2)
        zmin_2d = c_leg1.number_input("Min Strain (2D)", value=-50)
        zmax_2d = c_leg2.number_input("Max Strain (2D)", value=50)

    if st.button("Generate Map(s)", type="primary"):
        if sensor_locs is None:
            st.error("Metadata (Sensor Locations) is missing. Cannot map meters to indices.")
            st.stop()

        with st.spinner("Calculating Difference Profile..."):
            rows = (
                pl.scan_parquet(selected_file)
                .with_row_index("idx")
                .filter(pl.col("idx").is_in([idx_a, idx_b]))
                .collect()
            )
            
            try:
                row_a_data = rows.filter(pl.col("idx") == idx_a)["Data_Array"][0].to_numpy()
                row_b_data = rows.filter(pl.col("idx") == idx_b)["Data_Array"][0].to_numpy()
                diff_profile = row_a_data - row_b_data
            except Exception as e:
                st.error(f"Error extracting data rows: {e}")
                st.stop()

        # Configuration: (Label, Center, Plot_X, Offset_Minus, Offset_Plus, Should_Reverse)
        configs = [
            ("a", loc_a, 2.0, val_x, val_y, False), 
            ("b", loc_b, 1.5, val_y, val_x, True),  
            ("c", loc_c, 1.0, val_x, val_y, False), 
            ("d", loc_d, 0.5, val_y, val_x, True),  
            ("e", loc_e, 0.0, val_x, val_y, False), 
        ]

        plot_data_3d = []
        heatmap_z = []
        y_vals_common = None

        for label, center, plot_x, minus_off, plus_off, reverse_flag in configs:
            abs_start_m = center - minus_off
            abs_end_m   = center + plus_off
            
            idx_start = (np.abs(sensor_locs - abs_start_m)).argmin()
            idx_end   = (np.abs(sensor_locs - abs_end_m)).argmin()
            i_min, i_max = min(idx_start, idx_end), max(idx_start, idx_end)
            
            # Slice Data
            segment_z = diff_profile[i_min : i_max+1]
            
            # Reverse if needed
            if reverse_flag:
                segment_z = segment_z[::-1]
            
            # Standardize Y-Axis
            y_vals_standard = np.linspace(-val_x, val_y, len(segment_z))
            
            # Store for 3D
            if show_3d:
                x_vals_3d = [plot_x] * len(segment_z)
                plot_data_3d.append(go.Scatter3d(
                    x=x_vals_3d, y=y_vals_standard, z=segment_z,
                    mode='lines', name=f"Line {label} ({center}m)",
                    line=dict(width=6),
                    hovertemplate=f"<b>Line {label}</b><br>Grid Y: %{{y:.3f}}m<br>Diff: %{{z:.2f}}µε<extra></extra>"
                ))

            # Store for 2D Heatmap (Interpolation)
            if show_2d:
                target_len = 200 # Fixed resolution for stacking
                segment_resampled = np.interp(
                    np.linspace(0, 1, target_len), 
                    np.linspace(0, 1, len(segment_z)), 
                    segment_z
                )
                heatmap_z.append(segment_resampled)
                if y_vals_common is None:
                    y_vals_common = np.linspace(-val_x, val_y, target_len)

        # --- PLOT 3D ---
        if show_3d:
            range_x = 2.5 
            range_y = (val_x + val_y) * 1.2
            max_range = max(range_x, range_y)
            
            fig_3d = go.Figure(data=plot_data_3d)
            fig_3d.update_layout(
                title=f"3D Map ({all_times[idx_a]} vs {all_times[idx_b]})",
                scene=dict(
                    xaxis=dict(title="Section", tickvals=[0,0.5,1,1.5,2], ticktext=["e","d","c","b","a"], range=[-0.2, 2.2]),
                    yaxis=dict(title="Relative Distance (m)", range=[-val_x*1.1, val_y*1.1]),
                    zaxis=dict(title="Strain Diff (με)"),
                    camera=dict(eye=dict(x=1.5, y=-1.5, z=0.5)),
                    aspectmode='manual',
                    aspectratio=dict(x=2.2/max_range, y=range_y/max_range, z=0.5) 
                ),
                margin=dict(l=0, r=0, b=0, t=40),
                height=700
            )
            st.plotly_chart(fig_3d, use_container_width=True)

        # --- PLOT 2D HEATMAP ---
        if show_2d and heatmap_z:
            heatmap_matrix = np.array(heatmap_z[::-1]).T 
            
            fig_2d = go.Figure(data=go.Heatmap(
                z=heatmap_matrix,
                x=[0, 0.5, 1.0, 1.5, 2.0], # e, d, c, b, a
                y=y_vals_common,
                colorscale='RdBu_r',
                zsmooth='best',
                zmin=zmin_2d,
                zmax=zmax_2d,
                colorbar=dict(title="Strain Diff")
            ))
            
            fig_2d.update_layout(
                title="2D Interpolated Heatmap (To Scale)",
                xaxis=dict(
                    title="Section", 
                    tickvals=[0, 0.5, 1, 1.5, 2], 
                    ticktext=["e (0.0)", "d (0.5)", "c (1.0)", "b (1.5)", "a (2.0)"]
                ),
                yaxis=dict(
                    title="Relative Distance (m)",
                    scaleanchor="x", # Locks aspect ratio 1:1
                    scaleratio=1
                ),
                height=600
            )
            st.plotly_chart(fig_2d, use_container_width=True)

# ==========================================
# TAB 5: ANIMATION
# ==========================================
with tab5:
    st.write("### 🎬 Strain Animation (Profile vs Time)")
    st.caption("Exported as GIF, which inserts directly into PowerPoint and plays during slideshow.")

    c1, c2 = st.columns(2)
    t1_a = c1.text_input("Start Time (Animation)", value=str(start_ts), key="anim_start")
    t2_a = c2.text_input("End Time (Animation)", value=str(end_ts), key="anim_end")
    t2_qa = t2_a + ".999999" if len(t2_a) == 19 else t2_a

    c3, c4 = st.columns(2)
    anim_min = c3.number_input("Length Start (m)", value=min_m_def, step=0.1, key="anim_min")
    anim_max = c4.number_input("Length End (m)", value=max_m_def, step=0.1, key="anim_max")

    x_dir = st.radio("X-axis Orientation", ["Small → Large", "Large → Small"], horizontal=True)

    c5, c6 = st.columns(2)
    fps = c5.slider("FPS", min_value=1, max_value=20, value=8)
    max_frames = c6.slider("Max Frames (auto-downsample)", min_value=50, max_value=1200, value=300, step=50)

    use_manual_ylim = st.checkbox("Manual Y-axis range", value=False, key="anim_manual_ylim")
    y_min, y_max = None, None
    if use_manual_ylim:
        y1, y2 = st.columns(2)
        y_min = y1.number_input("Y Min (Strain)", value=-50.0)
        y_max = y2.number_input("Y Max (Strain)", value=50.0)

    if st.button("Generate Animation (GIF)", type="primary"):
        with st.spinner("Building animation..."):
            slice_df = (
                pl.scan_parquet(selected_file)
                .filter((pl.col("Time") >= t1_a) & (pl.col("Time") <= t2_qa))
                .collect()
            )

            if len(slice_df) == 0:
                st.warning("No data found for selected period.")
            else:
                full_matrix = np.vstack(slice_df["Data_Array"].to_numpy())
                all_locs = _build_spatial_axis(sensor_locs, full_matrix.shape[1])

                idx_start_a = (np.abs(all_locs - anim_min)).argmin()
                idx_end_a = (np.abs(all_locs - anim_max)).argmin()
                if idx_start_a > idx_end_a:
                    idx_start_a, idx_end_a = idx_end_a, idx_start_a
                if idx_start_a == idx_end_a:
                    idx_end_a = min(full_matrix.shape[1] - 1, idx_end_a + 1)

                safe_end = min(full_matrix.shape[1], idx_end_a + 1)
                anim_matrix = full_matrix[:, idx_start_a:safe_end]
                x_vals = all_locs[idx_start_a:safe_end]
                time_labels = slice_df["Time"].to_list()

                if len(time_labels) > max_frames:
                    skip = int(np.ceil(len(time_labels) / max_frames))
                    anim_matrix = anim_matrix[::skip, :]
                    time_labels = time_labels[::skip]
                else:
                    skip = 1

                reverse_x = (x_dir == "Large → Small")

                try:
                    gif_data = create_strain_animation_gif(
                        strain_matrix=anim_matrix,
                        x_values=x_vals,
                        time_labels=time_labels,
                        reverse_x=reverse_x,
                        fps=fps,
                        y_min=y_min,
                        y_max=y_max
                    )

                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_name = f"ODiSI_Strain_Animation_{ts}.gif"
                    with open(out_name, "wb") as f:
                        f.write(gif_data)

                    st.success(f"Saved: {out_name}")
                    st.caption(f"Frames used: {len(time_labels)} (downsample step={skip})")
                    st.download_button(
                        label="📥 Download GIF",
                        data=gif_data,
                        file_name=out_name,
                        mime="image/gif"
                    )
                except Exception as e:
                    st.error(f"Animation failed: {e}")

# ==========================================
# TAB 6: VERIFY
# ==========================================
with tab6:
    if sensor_locs is not None:
        st.dataframe(pd.DataFrame({"Idx": range(len(sensor_locs)), "Loc": sensor_locs}), width=600)
    else:
        st.warning("No metadata.")
