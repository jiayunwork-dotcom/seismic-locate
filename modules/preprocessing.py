import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import Stream, Trace
from scipy.signal import butter, filtfilt, detrend
import warnings
warnings.filterwarnings('ignore')


def preprocessing_page():
    st.title("⚙️ 波形预处理")
    st.markdown("---")

    if st.session_state.waveforms is None:
        st.warning("请先导入波形数据")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("预处理步骤")
        
        do_detrend = st.checkbox("去均值 (消除直流偏移)", value=True)
        do_demean = st.checkbox("去趋势 (线性拟合去除)", value=True)
        
        do_filter = st.checkbox("带通滤波", value=True)
        if do_filter:
            filter_order = st.slider("滤波器阶数", 1, 8, 4)
            freq_min = st.number_input("低频截止频率 (Hz)", value=1.0, min_value=0.01, step=0.1)
            freq_max = st.number_input("高频截止频率 (Hz)", value=10.0, min_value=0.1, step=0.1)
        
        do_response = st.checkbox("仪器响应去除", value=False)
        if do_response:
            output_type = st.selectbox("输出类型", ["位移", "速度", "加速度"], index=0)
        
        do_resample = st.checkbox("重采样", value=False)
        if do_resample:
            target_sr = st.number_input("目标采样率 (Hz)", value=50.0, min_value=1.0, step=1.0)

        if st.button("🔧 执行预处理", type="primary", use_container_width=True):
            with st.spinner("正在处理波形..."):
                processed = preprocess_waveforms(
                    st.session_state.waveforms,
                    do_detrend,
                    do_demean,
                    do_filter, filter_order, freq_min, freq_max,
                    do_response, output_type if do_response else None,
                    do_resample, target_sr if do_resample else None
                )
                st.session_state.processed_waveforms = processed
                st.success("预处理完成！")

    with col2:
        if st.session_state.processed_waveforms is not None:
            display_preprocessing_comparison()
        else:
            st.info("👈 配置参数后点击执行预处理")


def preprocess_waveforms(waveforms, do_detrend, do_demean, do_filter, filter_order, freq_min, freq_max, do_response, output_type, do_resample, target_sr):
    processed = []
    for tr in waveforms:
        tr_proc = tr.copy()
        
        if do_detrend:
            tr_proc.detrend('demean')
        
        if do_demean:
            tr_proc.detrend('linear')
        
        if do_filter:
            sr = tr_proc.stats.sampling_rate
            nyquist = sr / 2
            low = freq_min / nyquist
            high = freq_max / nyquist
            b, a = butter(filter_order, [low, high], btype='band')
            tr_proc.data = filtfilt(b, a, tr_proc.data)
        
        if do_response:
            pass
        
        if do_resample:
            tr_proc.resample(target_sr)
        
        processed.append(tr_proc)
    
    return processed


def display_preprocessing_comparison():
    st.subheader("📊 预处理前后对比")
    
    stations = sorted(set(tr.stats.station for tr in st.session_state.waveforms))
    selected_station = st.selectbox("选择台站", stations, key="preproc_sta")
    
    col_z_orig = None
    col_z_proc = None
    
    for tr in st.session_state.waveforms:
        if tr.stats.station == selected_station and tr.stats.channel[-1] == 'Z':
            col_z_orig = tr
            break
    
    for tr in st.session_state.processed_waveforms:
        if tr.stats.station == selected_station and tr.stats.channel[-1] == 'Z':
            col_z_proc = tr
            break
    
    if col_z_orig is not None and col_z_proc is not None:
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        
        times_orig = col_z_orig.times()
        times_proc = col_z_proc.times()
        
        axes[0].plot(times_orig, col_z_orig.data, 'k-', linewidth=0.5)
        axes[0].set_ylabel('原始振幅')
        axes[0].set_title(f'台站 {selected_station} - Z分量')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(times_proc, col_z_proc.data, 'r-', linewidth=0.5)
        axes[1].set_ylabel('处理后振幅')
        axes[1].set_xlabel('时间 (s)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        st.subheader("📈 频谱对比")
        fig, ax = plt.subplots(figsize=(10, 4))
        
        nfft_orig = len(col_z_orig.data)
        nfft_proc = len(col_z_proc.data)
        freq_orig = np.fft.rfftfreq(nfft_orig, d=1.0/col_z_orig.stats.sampling_rate)
        freq_proc = np.fft.rfftfreq(nfft_proc, d=1.0/col_z_proc.stats.sampling_rate)
        spec_orig = np.abs(np.fft.rfft(col_z_orig.data))
        spec_proc = np.abs(np.fft.rfft(col_z_proc.data))
        
        ax.loglog(freq_orig, spec_orig, 'k-', label='原始', alpha=0.7)
        ax.loglog(freq_proc, spec_proc, 'r-', label='处理后', alpha=0.7)
        ax.set_xlabel('频率 (Hz)')
        ax.set_ylabel('振幅谱')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
