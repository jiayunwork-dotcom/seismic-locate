import streamlit as st
import os
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import read, UTCDateTime
from obspy.io.sac import SACTrace
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')


def data_import_page():
    st.title("📡 数据导入")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("波形数据导入")
        file_type = st.radio("文件格式", ["MiniSEED", "SAC"], horizontal=True)
        
        uploaded_files = st.file_uploader(
            "上传波形文件（支持批量）",
            type=['mseed', 'msd', 'sac', 'SAC'] if file_type == "MiniSEED" else ['sac', 'SAC'],
            accept_multiple_files=True
        )

        st.subheader("台站元数据")
        meta_uploaded = st.file_uploader(
            "上传台站元数据文件 (CSV)",
            type=['csv']
        )

        if st.button("📥 导入数据", type="primary", use_container_width=True):
            if uploaded_files:
                with st.spinner("正在读取波形数据..."):
                    waveforms = load_waveforms(uploaded_files, file_type)
                    if waveforms:
                        st.session_state.waveforms = waveforms
                        st.session_state.processed_waveforms = None
                        st.success(f"成功导入 {len(waveforms)} 条波形记录")
                    else:
                        st.error("波形数据读取失败")
            else:
                st.warning("请先上传波形文件")

            if meta_uploaded:
                with st.spinner("正在读取台站元数据..."):
                    metadata = load_station_metadata(meta_uploaded)
                    if metadata is not None:
                        st.session_state.station_metadata = metadata
                        st.success(f"成功导入 {len(metadata)} 个台站元数据")

    with col2:
        if st.session_state.waveforms is not None:
            display_waveform_info(st.session_state.waveforms)
            st.markdown("---")
            display_station_map()
        else:
            st.info("👈 请先导入波形数据")

        if st.session_state.station_metadata is not None:
            st.markdown("---")
            display_metadata_table()


def load_waveforms(uploaded_files, file_type):
    st_list = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mseed' if file_type == "MiniSEED" else '.sac') as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        
        try:
            st_tmp = read(tmp_path)
            st_list.extend(st_tmp)
        except Exception as e:
            st.error(f"读取文件 {uploaded_file.name} 失败: {e}")
        finally:
            os.unlink(tmp_path)
    
    return st_list if st_list else None


def load_station_metadata(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        required_cols = ['station', 'longitude', 'latitude']
        if not all(col in df.columns for col in required_cols):
            st.error(f"CSV文件缺少必要列。需要: {required_cols}")
            return None
        return df
    except Exception as e:
        st.error(f"读取元数据失败: {e}")
        return None


def display_waveform_info(waveforms):
    st.subheader("📊 波形数据概览")
    
    stations = set()
    components = set()
    sampling_rates = set()
    start_times = []
    end_times = []
    
    for tr in waveforms:
        stations.add(tr.stats.station)
        components.add(tr.stats.channel[-1])
        sampling_rates.add(tr.stats.sampling_rate)
        start_times.append(tr.stats.starttime)
        end_times.append(tr.stats.endtime)
    
    info_df = pd.DataFrame({
        "项目": ["台站数量", "分量", "采样率 (Hz)", "起始时间", "结束时间", "记录条数"],
        "值": [
            len(stations),
            ', '.join(sorted(components)),
            ', '.join([str(sr) for sr in sorted(sampling_rates)]),
            str(min(start_times)),
            str(max(end_times)),
            len(waveforms)
        ]
    })
    
    st.table(info_df)
    
    st.subheader("📈 波形预览")
    station_list = sorted(stations)
    selected_station = st.selectbox("选择台站", station_list)
    
    z_tr = None
    for tr in waveforms:
        if tr.stats.station == selected_station and tr.stats.channel[-1] == 'Z':
            z_tr = tr
            break
    
    if z_tr is not None:
        fig, ax = plt.subplots(figsize=(10, 3))
        times = z_tr.times()
        ax.plot(times, z_tr.data, 'k-', linewidth=0.5)
        ax.set_xlabel('时间 (s)')
        ax.set_ylabel('振幅')
        ax.set_title(f'台站 {selected_station} - Z分量')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)


def display_station_map():
    st.subheader("🗺️ 台站分布地图")
    
    lats = []
    lons = []
    names = []
    
    if st.session_state.station_metadata is not None:
        meta = st.session_state.station_metadata
        lats = meta['latitude'].tolist()
        lons = meta['longitude'].tolist()
        names = meta['station'].tolist()
    else:
        coords = extract_station_coords_from_waveforms(st.session_state.waveforms)
        if coords:
            for sta, (lat, lon) in coords.items():
                lats.append(lat)
                lons.append(lon)
                names.append(sta)
    
    if lats and lons:
        fig = go.Figure()
        
        fig.add_trace(go.Scattermap(
            lat=lats,
            lon=lons,
            mode='markers+text',
            marker=dict(size=10, color='red', opacity=0.8),
            text=names,
            textposition="top center",
            name='台站'
        ))
        
        fig.update_layout(
            map=dict(
                style="open-street-map",
                zoom=5,
                center=dict(lat=np.mean(lats), lon=np.mean(lons))
            ),
            showlegend=False,
            height=400,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("无法获取台站坐标信息，请上传台站元数据")


def extract_station_coords_from_waveforms(waveforms):
    coords = {}
    for tr in waveforms:
        sta = tr.stats.station
        if sta not in coords:
            if hasattr(tr.stats, 'sac'):
                if hasattr(tr.stats.sac, 'stla') and hasattr(tr.stats.sac, 'stlo'):
                    coords[sta] = (tr.stats.sac.stla, tr.stats.sac.stlo)
    return coords


def display_metadata_table():
    st.subheader("📋 台站元数据表")
    st.dataframe(st.session_state.station_metadata, use_container_width=True, height=200)
