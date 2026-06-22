import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import UTCDateTime
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings('ignore')


def magnitude_page():
    st.title("📏 震级计算")
    st.markdown("---")

    waveforms = st.session_state.processed_waveforms or st.session_state.waveforms
    if waveforms is None:
        st.warning("请先导入波形数据")
        return
    if not st.session_state.picks:
        st.warning("请先进行震相拾取")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("震级类型")
        calc_ml = st.checkbox("ML 本地震级", value=True)
        calc_mw = st.checkbox("Mw 矩震级", value=True)
        
        if calc_ml:
            st.subheader("ML参数")
            ml_freq_min = st.number_input("Wood-Anderson低频(Hz)", value=1.0, min_value=0.1, step=0.1, key="ml_fmin")
            ml_freq_max = st.number_input("Wood-Anderson高频(Hz)", value=20.0, min_value=1.0, step=1.0, key="ml_fmax")
            attenuation = st.number_input("区域衰减系数", value=0.003, format="%.4f", step=0.0001)
        
        if calc_mw:
            st.subheader("Mw参数")
            p_window_len = st.number_input("P波窗口长度(s)", value=5.0, min_value=1.0, step=0.5)
            density = st.number_input("介质密度(kg/m³)", value=2700.0, step=100.0)
            shear_velocity = st.number_input("剪切波速(km/s)", value=3.5, step=0.1)
        
        if st.button("📊 计算震级", type="primary", use_container_width=True):
            with st.spinner("正在计算震级..."):
                mag_result = {}
                
                if calc_ml:
                    ml_result = calculate_ml(waveforms, ml_freq_min, ml_freq_max, attenuation)
                    mag_result['ML'] = ml_result
                
                if calc_mw:
                    mw_result = calculate_mw(waveforms, p_window_len, density, shear_velocity)
                    mag_result['Mw'] = mw_result
                
                st.session_state.magnitude_result = mag_result
                st.success("震级计算完成！")

    with col2:
        if st.session_state.magnitude_result:
            display_magnitude_results()
        else:
            st.info("👈 配置参数后点击计算震级")


def wood_anderson_response(data, sampling_rate):
    """模拟Wood-Anderson地震仪响应"""
    omega_0 = 2 * np.pi * 1.25
    h = 0.8
    
    freqs = np.fft.rfftfreq(len(data), d=1.0/sampling_rate)
    omega = 2 * np.pi * freqs
    omega[0] = 1e-10
    
    resp = omega**2 / (omega_0**2 - omega**2 + 2j * h * omega_0 * omega)
    resp[0] = 0
    
    fft_data = np.fft.rfft(data)
    fft_filtered = fft_data * resp
    filtered = np.fft.irfft(fft_filtered, n=len(data))
    
    return filtered


def calculate_ml(waveforms, freq_min, freq_max, attenuation):
    waveforms = st.session_state.processed_waveforms or st.session_state.waveforms
    picks = st.session_state.picks
    location = st.session_state.location_result
    
    ml_values = []
    station_data = []
    
    stations = set(tr.stats.station for tr in waveforms)
    
    for sta in stations:
        if sta not in picks or 'P' not in picks[sta]:
            continue
        
        z_tr = None
        for tr in waveforms:
            if tr.stats.station == sta and tr.stats.channel[-1] == 'Z':
                z_tr = tr
                break
        
        if z_tr is None:
            continue
        
        sr = z_tr.stats.sampling_rate
        data = z_tr.data.copy()
        
        nyquist = sr / 2
        low = freq_min / nyquist
        high = freq_max / nyquist
        b, a = butter(2, [low, high], btype='band')
        data = filtfilt(b, a, data)
        
        data_wa = wood_anderson_response(data, sr)
        
        max_amp = np.max(np.abs(data_wa))
        max_amp_mm = max_amp * 1000
        
        if location:
            lat, lon = get_station_coords_from_location(sta, location)
            if lat is not None and location['latitude'] is not None:
                dist_km = haversine_distance(location['latitude'], location['longitude'], lat, lon)
            else:
                dist_km = 100.0
        else:
            dist_km = 100.0
        
        ml_station = np.log10(max_amp_mm) + attenuation * dist_km + 1.0
        
        ml_values.append(ml_station)
        station_data.append({
            'station': sta,
            'max_amp_um': max_amp * 1e6,
            'distance_km': dist_km,
            'ML': ml_station
        })
    
    if not ml_values:
        return None
    
    ml_mean = np.mean(ml_values)
    ml_std = np.std(ml_values)
    
    return {
        'ML_mean': ml_mean,
        'ML_std': ml_std,
        'n_stations': len(ml_values),
        'stations': station_data
    }


def get_station_coords_from_location(station_name, location):
    if station_name in location.get('stations', []):
        idx = location['stations'].index(station_name)
        return location['station_lats'][idx], location['station_lons'][idx]
    return None, None


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c


def calculate_mw(waveforms, window_len, density, shear_vel):
    waveforms = st.session_state.processed_waveforms or st.session_state.waveforms
    picks = st.session_state.picks
    location = st.session_state.location_result
    
    m0_values = []
    station_data = []
    
    stations = set(tr.stats.station for tr in waveforms)
    
    for sta in stations:
        if sta not in picks or 'P' not in picks[sta]:
            continue
        
        z_tr = None
        for tr in waveforms:
            if tr.stats.station == sta and tr.stats.channel[-1] == 'Z':
                z_tr = tr
                break
        
        if z_tr is None:
            continue
        
        sr = z_tr.stats.sampling_rate
        p_time = picks[sta]['P']
        
        start_idx = int((p_time - z_tr.stats.starttime) * sr)
        end_idx = min(start_idx + int(window_len * sr), len(z_tr.data))
        
        if end_idx - start_idx < int(sr):
            continue
        
        window_data = z_tr.data[start_idx:end_idx]
        
        nfft = len(window_data)
        freqs = np.fft.rfftfreq(nfft, d=1.0/sr)
        spectrum = np.abs(np.fft.rfft(window_data * np.hanning(nfft)))
        
        low_freq_idx = np.where((freqs >= 0.1) & (freqs <= 1.0))[0]
        if len(low_freq_idx) > 0:
            low_freq_amp = np.mean(spectrum[low_freq_idx])
        else:
            continue
        
        corner_freq = estimate_corner_frequency(freqs, spectrum, low_freq_amp)
        
        if corner_freq is None or corner_freq <= 0:
            continue
        
        radiation_pattern = 0.52
        
        if location:
            lat, lon = get_station_coords_from_location(sta, location)
            if lat is not None and location['latitude'] is not None:
                dist_km = haversine_distance(location['latitude'], location['longitude'], lat, lon)
            else:
                dist_km = 100.0
        else:
            dist_km = 100.0
        
        dist_m = dist_km * 1000
        shear_vel_m = shear_vel * 1000
        
        low_freq_disp = low_freq_amp / (2 * np.pi * freqs[low_freq_idx[0]] if len(low_freq_idx) > 0 else 1)
        
        m0_station = (4 * np.pi * density * shear_vel_m**3 * dist_m * low_freq_disp) / radiation_pattern
        
        m0_station = low_freq_amp * 4 * np.pi * density * (shear_vel_m)**3 * dist_m / (radiation_pattern * 2 * np.pi * 1.0)
        
        m0_values.append(m0_station)
        station_data.append({
            'station': sta,
            'corner_freq_Hz': corner_freq,
            'low_freq_amp': low_freq_amp,
            'M0_Nm': m0_station
        })
    
    if not m0_values:
        return None
    
    m0_mean = np.mean(m0_values)
    
    mw = (2.0 / 3.0) * (np.log10(m0_mean) - 9.1)
    
    return {
        'Mw': mw,
        'M0': m0_mean,
        'n_stations': len(m0_values),
        'stations': station_data
    }


def estimate_corner_frequency(freqs, spectrum, low_freq_amp):
    if low_freq_amp <= 0:
        return None
    
    target_amp = low_freq_amp / (2**0.5)
    
    for i in range(1, len(freqs)):
        if spectrum[i] < target_amp and spectrum[i-1] >= target_amp:
            f1 = freqs[i-1]
            f2 = freqs[i]
            a1 = spectrum[i-1]
            a2 = spectrum[i]
            frac = (target_amp - a1) / (a2 - a1) if a2 != a1 else 0
            return f1 + frac * (f2 - f1)
    
    return freqs[len(freqs) // 2] if len(freqs) > 0 else 1.0


def display_magnitude_results():
    result = st.session_state.magnitude_result
    
    st.subheader("🎯 震级结果摘要")
    
    cols = st.columns(2)
    
    if 'ML' in result and result['ML']:
        ml = result['ML']
        with cols[0]:
            st.metric("ML 本地震级", f"{ml['ML_mean']:.2f} ± {ml['ML_std']:.2f}")
            st.caption(f"基于 {ml['n_stations']} 个台站")
    
    if 'Mw' in result and result['Mw']:
        mw = result['Mw']
        with cols[1]:
            st.metric("Mw 矩震级", f"{mw['Mw']:.2f}")
            st.caption(f"标量地震矩 M₀ = {mw['M0']:.2e} N·m")
    
    if 'ML' in result and result['ML']:
        st.subheader("📊 ML 各台站详情")
        ml_df = pd.DataFrame(result['ML']['stations'])
        ml_df.columns = ['台站', '最大振幅(μm)', '震中距(km)', 'ML']
        st.dataframe(ml_df, use_container_width=True, hide_index=True)
        
        st.subheader("📈 ML 分布")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist([s['ML'] for s in result['ML']['stations']], bins=10, edgecolor='black', alpha=0.7)
        ax.axvline(result['ML']['ML_mean'], color='r', linestyle='--', label=f'均值={result["ML"]["ML_mean"]:.2f}')
        ax.set_xlabel('ML')
        ax.set_ylabel('台站数')
        ax.set_title('各台站ML分布')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
    
    if 'Mw' in result and result['Mw']:
        st.subheader("📊 Mw 各台站详情")
        mw_df = pd.DataFrame(result['Mw']['stations'])
        mw_df.columns = ['台站', '拐角频率(Hz)', '低频振幅', 'M₀(N·m)']
        st.dataframe(mw_df, use_container_width=True, hide_index=True)
