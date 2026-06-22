import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import UTCDateTime
import warnings
warnings.filterwarnings('ignore')


def phase_picking_page():
    st.title("🔍 震相自动拾取")
    st.markdown("---")

    waveforms = st.session_state.processed_waveforms or st.session_state.waveforms
    if waveforms is None:
        st.warning("请先导入波形数据")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("P波拾取参数")
        
        p_method = st.radio("P波拾取算法", ["STA/LTA", "AIC", "STA/LTA + AIC"], index=2)
        
        sta_len = st.number_input("短时窗长度 (s)", value=1.0, min_value=0.1, step=0.1)
        lta_len = st.number_input("长时窗长度 (s)", value=10.0, min_value=1.0, step=0.5)
        trigger_on = st.number_input("触发阈值", value=3.0, min_value=1.0, step=0.5)
        trigger_off = st.number_input("解除阈值", value=1.5, min_value=0.5, step=0.5)
        
        st.subheader("S波拾取参数")
        do_s_pick = st.checkbox("拾取S波", value=True)
        if do_s_pick:
            s_search_window = st.number_input("S波搜索窗口 (P波后秒数)", value=30.0, min_value=5.0, step=5.0)
        
        stations = sorted(set(tr.stats.station for tr in waveforms))
        selected_stations = st.multiselect("选择台站", stations, default=stations[:3] if len(stations) > 3 else stations)

        if st.button("🎯 执行拾取", type="primary", use_container_width=True):
            with st.spinner("正在进行震相拾取..."):
                picks = {}
                for sta in selected_stations:
                    z_tr = None
                    n_tr = None
                    e_tr = None
                    for tr in waveforms:
                        if tr.stats.station == sta:
                            comp = tr.stats.channel[-1]
                            if comp == 'Z':
                                z_tr = tr
                            elif comp in ['N', '1']:
                                n_tr = tr
                            elif comp in ['E', '2']:
                                e_tr = tr
                    
                    if z_tr is not None:
                        p_time, p_confidence = pick_p_wave(z_tr, sta_len, lta_len, trigger_on, trigger_off, p_method)
                        picks[sta] = {'P': p_time, 'P_confidence': p_confidence}
                        
                        if do_s_pick and (n_tr is not None or e_tr is not None):
                            s_time, s_confidence = pick_s_wave(z_tr, n_tr, e_tr, p_time, s_search_window, sta_len, lta_len, trigger_on)
                            picks[sta]['S'] = s_time
                            picks[sta]['S_confidence'] = s_confidence
                
                st.session_state.picks = picks
                st.success(f"完成 {len(picks)} 个台站的震相拾取")

    with col2:
        if st.session_state.picks:
            display_picking_results(waveforms)
        else:
            st.info("👈 配置参数后点击执行拾取")


def sta_lta(data, sampling_rate, sta_len, lta_len):
    nsta = int(sta_len * sampling_rate)
    nlta = int(lta_len * sampling_rate)
    
    if nsta < 1 or nlta < nsta:
        return np.zeros_like(data)
    
    energy = data ** 2
    
    sta = np.zeros_like(energy)
    lta = np.zeros_like(energy)
    
    cumsum = np.cumsum(energy)
    
    for i in range(nlta, len(energy)):
        if i >= nsta:
            sta[i] = (cumsum[i] - cumsum[i - nsta]) / nsta
        lta[i] = (cumsum[i] - cumsum[i - nlta]) / nlta
    
    ratio = np.zeros_like(energy)
    valid = lta > 0
    ratio[valid] = sta[valid] / lta[valid]
    
    return ratio


def aic_picker(data, sampling_rate, t_start_idx, t_end_idx):
    start_idx = max(0, t_start_idx)
    end_idx = min(len(data), t_end_idx)
    
    if end_idx - start_idx < 10:
        return (start_idx + end_idx) // 2
    
    window_data = data[start_idx:end_idx]
    n = len(window_data)
    
    aic = np.zeros(n)
    
    for k in range(1, n - 1):
        var1 = np.var(window_data[:k]) if k > 1 else 1e-10
        var2 = np.var(window_data[k:]) if n - k > 1 else 1e-10
        aic[k] = k * np.log(var1 + 1e-10) + (n - k) * np.log(var2 + 1e-10)
    
    min_idx = np.argmin(aic[1:-1]) + 1
    
    return start_idx + min_idx


def pick_p_wave(trace, sta_len, lta_len, trigger_on, trigger_off, method):
    sr = trace.stats.sampling_rate
    data = trace.data
    
    ratio = sta_lta(data, sr, sta_len, lta_len)
    
    triggered = False
    trigger_idx = None
    
    for i in range(len(ratio)):
        if not triggered and ratio[i] > trigger_on:
            triggered = True
            trigger_idx = i
        elif triggered and ratio[i] < trigger_off:
            triggered = False
    
    if trigger_idx is None:
        max_idx = np.argmax(ratio)
        trigger_idx = max_idx - int(sta_len * sr)
        trigger_idx = max(0, trigger_idx)
    
    if "AIC" in method:
        search_start = max(0, trigger_idx - int(2 * sta_len * sr))
        search_end = min(len(data), trigger_idx + int(2 * sta_len * sr))
        aic_idx = aic_picker(data, sr, search_start, search_end)
        pick_idx = aic_idx
        confidence = min(1.0, ratio[trigger_idx] / trigger_on)
    else:
        pick_idx = trigger_idx
        confidence = min(1.0, ratio[trigger_idx] / trigger_on)
    
    pick_time = trace.stats.starttime + pick_idx / sr
    
    return pick_time, confidence


def pick_s_wave(z_tr, n_tr, e_tr, p_time, search_window, sta_len, lta_len, trigger_on):
    if n_tr is None and e_tr is None:
        return None, 0
    
    sr = z_tr.stats.sampling_rate
    
    if n_tr is not None and e_tr is not None:
        n_data = n_tr.data
        e_data = e_tr.data
        min_len = min(len(n_data), len(e_data))
        transverse = np.sqrt(n_data[:min_len]**2 + e_data[:min_len]**2)
        start_time = min(n_tr.stats.starttime, e_tr.stats.starttime)
    elif n_tr is not None:
        transverse = n_tr.data.copy()
        start_time = n_tr.stats.starttime
    else:
        transverse = e_tr.data.copy()
        start_time = e_tr.stats.starttime
    
    p_offset = p_time - start_time
    p_idx = int(p_offset * sr)
    
    search_start = max(0, p_idx + int(1.0 * sr))
    search_end = min(len(transverse), p_idx + int(search_window * sr))
    
    if search_end - search_start < 10:
        return None, 0
    
    search_data = transverse[search_start:search_end]
    ratio = sta_lta(search_data, sr, sta_len, lta_len)
    
    max_ratio_idx = np.argmax(ratio)
    s_idx_in_window = max_ratio_idx
    
    s_idx = search_start + s_idx_in_window
    s_time = start_time + s_idx / sr
    confidence = min(1.0, ratio[max_ratio_idx] / trigger_on)
    
    return s_time, confidence


def display_picking_results(waveforms):
    st.subheader("📊 拾取结果")
    
    picks = st.session_state.picks
    
    pick_data = []
    for sta, p in picks.items():
        p_time = str(p.get('P', 'N/A'))
        p_conf = f"{p.get('P_confidence', 0):.2f}" if 'P_confidence' in p else 'N/A'
        s_time = str(p.get('S', 'N/A')) if 'S' in p and p['S'] else 'N/A'
        s_conf = f"{p.get('S_confidence', 0):.2f}" if 'S_confidence' in p else 'N/A'
        pick_data.append({
            '台站': sta,
            'P波到时': p_time,
            'P波置信度': p_conf,
            'S波到时': s_time,
            'S波置信度': s_conf
        })
    
    df = pd.DataFrame(pick_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.subheader("📈 波形显示")
    stations = list(picks.keys())
    selected_station = st.selectbox("选择台站查看详情", stations, key="pick_sta")
    
    z_tr = None
    n_tr = None
    e_tr = None
    for tr in waveforms:
        if tr.stats.station == selected_station:
            comp = tr.stats.channel[-1]
            if comp == 'Z':
                z_tr = tr
            elif comp in ['N', '1']:
                n_tr = tr
            elif comp in ['E', '2']:
                e_tr = tr
    
    if z_tr is not None:
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        
        times_z = z_tr.times()
        axes[0].plot(times_z, z_tr.data, 'k-', linewidth=0.5)
        axes[0].set_ylabel('Z 振幅')
        axes[0].set_title(f'台站 {selected_station} - 三分量波形')
        axes[0].grid(True, alpha=0.3)
        
        if 'P' in picks[selected_station]:
            p_time = picks[selected_station]['P']
            p_offset = p_time - z_tr.stats.starttime
            axes[0].axvline(p_offset, color='r', linestyle='--', label='P波')
            axes[0].legend()
        
        if n_tr is not None:
            times_n = n_tr.times()
            axes[1].plot(times_n, n_tr.data, 'b-', linewidth=0.5)
            axes[1].set_ylabel('N 振幅')
            axes[1].grid(True, alpha=0.3)
        
        if e_tr is not None:
            times_e = e_tr.times()
            axes[2].plot(times_e, e_tr.data, 'g-', linewidth=0.5)
            axes[2].set_ylabel('E 振幅')
            axes[2].set_xlabel('时间 (s)')
            axes[2].grid(True, alpha=0.3)
        
        if 'S' in picks[selected_station] and picks[selected_station]['S']:
            s_time = picks[selected_station]['S']
            ref_tr = n_tr if n_tr is not None else z_tr
            s_offset = s_time - ref_tr.stats.starttime
            for ax in axes[1:]:
                ax.axvline(s_offset, color='orange', linestyle='--', label='S波')
            axes[1].legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
