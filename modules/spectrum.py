import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from obspy import UTCDateTime
import warnings
warnings.filterwarnings('ignore')


def spectrum_page():
    st.title("📊 频谱分析")
    st.markdown("---")

    waveforms = st.session_state.processed_waveforms or st.session_state.waveforms
    if waveforms is None:
        st.warning("请先导入波形数据")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("分析参数")
        
        stations = sorted(set(tr.stats.station for tr in waveforms))
        selected_stations = st.multiselect("选择台站", stations, default=stations[:1] if stations else [])
        
        components = ['Z', 'N', 'E']
        selected_comp = st.selectbox("选择分量", components, index=0)
        
        spectrum_type = st.radio("频谱类型", ["振幅谱", "相位谱", "功率谱密度"], index=0)
        
        use_welch = st.checkbox("使用Welch方法降噪", value=True)
        if use_welch:
            nperseg = st.number_input("每段长度(样本数)", value=256, min_value=64, step=64)
            noverlap = st.number_input("重叠长度(样本数)", value=128, min_value=32, step=32)
        
        window_type = st.selectbox("窗函数", ["汉宁窗", "汉明窗", "布莱克曼窗", "无"], index=0)
        
        if st.button("📈 分析频谱", type="primary", use_container_width=True):
            with st.spinner("正在计算频谱..."):
                spec_results = analyze_spectrum(
                    waveforms, selected_stations, selected_comp,
                    spectrum_type, use_welch, nperseg, noverlap, window_type
                )
                st.session_state.spec_results = spec_results
                st.success("频谱分析完成！")

    with col2:
        if hasattr(st.session_state, 'spec_results') and st.session_state.spec_results:
            display_spectrum_results()
        else:
            st.info("👈 配置参数后点击分析频谱")


def analyze_spectrum(waveforms, stations, component, spec_type, use_welch, nperseg, noverlap, window_type):
    results = []
    
    window_map = {
        "汉宁窗": np.hanning,
        "汉明窗": np.hamming,
        "布莱克曼窗": np.blackman,
        "无": None
    }
    window_func = window_map.get(window_type, None)
    
    for sta in stations:
        tr = None
        for t in waveforms:
            if t.stats.station == sta and t.stats.channel[-1] == component:
                tr = t
                break
        
        if tr is None:
            continue
        
        sr = tr.stats.sampling_rate
        data = tr.data.copy()
        
        if use_welch and spec_type == "功率谱密度":
            freqs, power = welch(data, fs=sr, nperseg=nperseg, noverlap=noverlap, window='hann')
            amp = np.sqrt(power)
            phase = np.zeros_like(freqs)
        else:
            n = len(data)
            if window_func:
                data = data * window_func(n)
            
            fft = np.fft.rfft(data)
            freqs = np.fft.rfftfreq(n, d=1.0/sr)
            
            amp = np.abs(fft)
            phase = np.angle(fft)
            power = (amp ** 2) / n
        
        if spec_type == "振幅谱":
            y_data = amp
        elif spec_type == "相位谱":
            y_data = phase
        else:
            y_data = power
        
        results.append({
            'station': sta,
            'freqs': freqs,
            'data': y_data,
            'type': spec_type
        })
    
    return results


def display_spectrum_results():
    results = st.session_state.spec_results
    
    st.subheader("📈 频谱图")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for res in results:
        freqs = res['freqs']
        data = res['data']
        
        if res['type'] in ['振幅谱', '功率谱密度']:
            ax.loglog(freqs, data, label=res['station'], alpha=0.7)
        else:
            ax.semilogx(freqs, data, label=res['station'], alpha=0.7)
    
    ax.set_xlabel('频率 (Hz)')
    ax.set_ylabel(results[0]['type'])
    ax.set_title(f"多台站{results[0]['type']}对比")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    plt.close(fig)
    
    st.subheader("⚡ 拐角频率与应力降")
    
    if st.button("计算拐角频率和应力降", type="secondary"):
        corner_results = calculate_stress_drop(results)
        display_stress_drop_results(corner_results)


def calculate_stress_drop(spec_results):
    results = []
    
    for res in spec_results:
        freqs = res['freqs']
        data = res['data']
        
        if res['type'] == '相位谱':
            continue
        
        low_freq_mask = (freqs >= 0.05) & (freqs <= 0.5)
        if not np.any(low_freq_mask):
            continue
        
        low_freq_data = data[low_freq_mask]
        low_level = np.mean(low_freq_data)
        
        corner_freq = find_corner_frequency(freqs, data, low_level)
        
        if corner_freq and corner_freq > 0:
            vs = 3500
            stress_drop = 7 * low_level * 1e-6 / (16 * corner_freq**3)
            
            results.append({
                'station': res['station'],
                'corner_freq': corner_freq,
                'low_level': low_level,
                'stress_drop': stress_drop
            })
    
    return results


def find_corner_frequency(freqs, spectrum, low_level):
    target = low_level / np.sqrt(2)
    
    above = spectrum > target
    below = spectrum <= target
    
    cross_idx = np.where(above & np.roll(below, 1))[0]
    
    if len(cross_idx) > 0:
        idx = cross_idx[0]
        if idx > 0 and idx < len(freqs):
            f1 = freqs[idx-1]
            f2 = freqs[idx]
            s1 = spectrum[idx-1]
            s2 = spectrum[idx]
            frac = (target - s1) / (s2 - s1) if s2 != s1 else 0
            return f1 + frac * (f2 - f1)
    
    return None


def display_stress_drop_results(corner_results):
    if not corner_results:
        st.warning("无法计算拐角频率")
        return
    
    st.subheader("📊 应力降计算结果")
    
    df = pd.DataFrame(corner_results)
    df.columns = ['台站', '拐角频率(Hz)', '低频水平', '应力降(Pa)']
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    results = st.session_state.spec_results
    
    for res in results:
        freqs = res['freqs']
        data = res['data']
        
        if res['type'] == '相位谱':
            continue
        
        ax.loglog(freqs, data, label=res['station'], alpha=0.6)
    
    for cres in corner_results:
        ax.axvline(cres['corner_freq'], color='red', linestyle='--', alpha=0.5)
        ax.scatter([cres['corner_freq']], [cres['low_level'] / np.sqrt(2)], 
                   color='red', s=50, zorder=5)
    
    ax.set_xlabel('频率 (Hz)')
    ax.set_ylabel('振幅')
    ax.set_title('振幅谱与拐角频率')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    plt.close(fig)
