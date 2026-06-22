import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import UTCDateTime
import warnings
warnings.filterwarnings('ignore')


def focal_mechanism_page():
    st.title("🌐 震源机制解")
    st.markdown("---")

    waveforms = st.session_state.processed_waveforms or st.session_state.waveforms
    if waveforms is None:
        st.warning("请先导入波形数据")
        return
    if not st.session_state.picks:
        st.warning("请先进行震相拾取")
        return
    if not st.session_state.location_result:
        st.warning("请先进行震源定位")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("分析参数")
        
        polarity_window = st.number_input("初动判断窗口(s)", value=0.5, min_value=0.1, step=0.1)
        
        st.subheader("格点搜索参数")
        strike_step = st.number_input("走向步长(°)", value=10, min_value=5, max_value=30)
        dip_step = st.number_input("倾角步长(°)", value=10, min_value=5, max_value=30)
        rake_step = st.number_input("滑动角步长(°)", value=10, min_value=5, max_value=30)
        
        if st.button("🔍 求解机制解", type="primary", use_container_width=True):
            with st.spinner("正在求解震源机制解..."):
                fm_result = solve_focal_mechanism(polarity_window, strike_step, dip_step, rake_step)
                if fm_result:
                    st.session_state.focal_mechanism = fm_result
                    st.success("震源机制解求解完成！")
                else:
                    st.error("求解失败，请检查数据")

    with col2:
        if st.session_state.focal_mechanism:
            display_focal_mechanism()
        else:
            st.info("👈 配置参数后点击求解机制解")


def get_polarities(waveforms, picks, location, window_len=0.5):
    polarities = []
    stations = []
    takeoff_angles = []
    azimuths = []
    
    for sta in picks.keys():
        if 'P' not in picks[sta]:
            continue
        
        z_tr = None
        for tr in waveforms:
            if tr.stats.station == sta and tr.stats.channel[-1] == 'Z':
                z_tr = tr
                break
        
        if z_tr is None:
            continue
        
        p_time = picks[sta]['P']
        sr = z_tr.stats.sampling_rate
        start_idx = int((p_time - z_tr.stats.starttime) * sr)
        end_idx = min(start_idx + int(window_len * sr), len(z_tr.data))
        
        if end_idx - start_idx < 3:
            continue
        
        window_data = z_tr.data[start_idx:end_idx]
        first_peak_idx = np.argmax(np.abs(window_data))
        first_peak_val = window_data[first_peak_idx]
        
        polarity = 1 if first_peak_val > 0 else -1
        
        if sta in location['stations']:
            idx = location['stations'].index(sta)
            sta_lat = location['station_lats'][idx]
            sta_lon = location['station_lons'][idx]
            
            epi_lat = location['latitude']
            epi_lon = location['longitude']
            depth = location['depth_km']
            
            azimuth = calculate_azimuth(epi_lat, epi_lon, sta_lat, sta_lon)
            takeoff_angle = calculate_takeoff_angle(epi_lat, epi_lon, depth, sta_lat, sta_lon)
            
            polarities.append(polarity)
            stations.append(sta)
            takeoff_angles.append(takeoff_angle)
            azimuths.append(azimuth)
    
    return {
        'stations': stations,
        'polarities': polarities,
        'takeoff_angles': takeoff_angles,
        'azimuths': azimuths
    }


def calculate_azimuth(lat1, lon1, lat2, lon2):
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    
    x = np.sin(dlon) * np.cos(lat2_rad)
    y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
    
    az = np.degrees(np.arctan2(x, y))
    return az % 360


def calculate_takeoff_angle(epi_lat, epi_lon, depth_km, sta_lat, sta_lon):
    from geopy.distance import geodesic
    
    dist_km = geodesic((epi_lat, epi_lon), (sta_lat, sta_lon)).kilometers
    
    takeoff = np.degrees(np.arctan2(dist_km, depth_km))
    
    return takeoff


def compute_p_for_nodal_plane(strike, dip, rake, takeoff_angle, azimuth):
    takeoff_rad = np.radians(takeoff_angle)
    az_rad = np.radians(azimuth)
    
    x = np.sin(takeoff_rad) * np.sin(az_rad)
    y = np.sin(takeoff_rad) * np.cos(az_rad)
    z = np.cos(takeoff_rad)
    
    strike_rad = np.radians(strike)
    dip_rad = np.radians(dip)
    rake_rad = np.radians(rake)
    
    n1 = np.sin(dip_rad) * np.sin(strike_rad)
    n2 = -np.sin(dip_rad) * np.cos(strike_rad)
    n3 = np.cos(dip_rad)
    
    d1 = np.cos(rake_rad) * np.cos(strike_rad) + np.sin(rake_rad) * np.cos(dip_rad) * np.sin(strike_rad)
    d2 = np.cos(rake_rad) * np.sin(strike_rad) - np.sin(rake_rad) * np.cos(dip_rad) * np.cos(strike_rad)
    d3 = -np.sin(rake_rad) * np.sin(dip_rad)
    
    amplitude = n1 * d1 * x**2 + n2 * d2 * y**2 + n3 * d3 * z**2 + \
                (n1 * d2 + n2 * d1) * x * y + \
                (n1 * d3 + n3 * d1) * x * z + \
                (n2 * d3 + n3 * d2) * y * z
    
    amplitude *= 2
    
    return np.sign(amplitude)


def solve_focal_mechanism(window_len, strike_step, dip_step, rake_step):
    waveforms = st.session_state.processed_waveforms or st.session_state.waveforms
    picks = st.session_state.picks
    location = st.session_state.location_result
    
    polarity_data = get_polarities(waveforms, picks, location, window_len)
    
    if len(polarity_data['stations']) < 4:
        st.error("极性数据不足，至少需要4个台站")
        return None
    
    best_misfit = len(polarity_data['stations']) + 1
    best_strike = 0
    best_dip = 45
    best_rake = 0
    
    for strike in range(0, 360, int(strike_step)):
        for dip in range(1, 90, int(dip_step)):
            for rake in range(0, 360, int(rake_step)):
                misfit = 0
                
                for i in range(len(polarity_data['stations'])):
                    takeoff = polarity_data['takeoff_angles'][i]
                    az = polarity_data['azimuths'][i]
                    obs_pol = polarity_data['polarities'][i]
                    
                    calc_pol = compute_p_for_nodal_plane(strike, dip, rake, takeoff, az)
                    
                    if calc_pol != 0 and calc_pol != obs_pol:
                        misfit += 1
                
                if misfit < best_misfit:
                    best_misfit = misfit
                    best_strike = strike
                    best_dip = dip
                    best_rake = rake
    
    second_plane = compute_auxiliary_plane(best_strike, best_dip, best_rake)
    
    return {
        'strike1': best_strike,
        'dip1': best_dip,
        'rake1': best_rake,
        'strike2': second_plane[0],
        'dip2': second_plane[1],
        'rake2': second_plane[2],
        'misfit': best_misfit,
        'n_stations': len(polarity_data['stations']),
        'polarity_data': polarity_data
    }


def compute_auxiliary_plane(strike, dip, rake):
    strike_rad = np.radians(strike)
    dip_rad = np.radians(dip)
    rake_rad = np.radians(rake)
    
    n1 = np.sin(dip_rad) * np.sin(strike_rad)
    n2 = -np.sin(dip_rad) * np.cos(strike_rad)
    n3 = np.cos(dip_rad)
    
    d1 = np.cos(rake_rad) * np.cos(strike_rad) + np.sin(rake_rad) * np.cos(dip_rad) * np.sin(strike_rad)
    d2 = np.cos(rake_rad) * np.sin(strike_rad) - np.sin(rake_rad) * np.cos(dip_rad) * np.cos(strike_rad)
    d3 = -np.sin(rake_rad) * np.sin(dip_rad)
    
    n2_strike = np.arctan2(d1, -d2)
    n2_dip = np.arccos(np.clip(d3, -1, 1))
    
    n1_vec = np.array([n1, n2, n3])
    d2_vec = np.array([-n2 * np.sin(n2_strike) - n1 * np.cos(n2_strike),
                        n1 * np.sin(n2_strike) - n2 * np.cos(n2_strike),
                        n3 * np.sin(n2_dip)])
    
    if n2_dip > np.pi / 2:
        n2_dip = np.pi - n2_dip
        n2_strike += np.pi
    
    n2_rake = np.arcsin(np.clip(np.dot(n1_vec, d2_vec), -1, 1))
    
    if n3 < 0:
        n2_rake = np.pi - n2_rake
    
    return (np.degrees(n2_strike) % 360, np.degrees(n2_dip), np.degrees(n2_rake))


def display_focal_mechanism():
    fm = st.session_state.focal_mechanism
    
    st.subheader("🎯 最佳双力偶解")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**节面 I**")
        st.metric("走向", f"{fm['strike1']:.1f}°")
        st.metric("倾角", f"{fm['dip1']:.1f}°")
        st.metric("滑动角", f"{fm['rake1']:.1f}°")
    
    with col2:
        st.info("**节面 II**")
        st.metric("走向", f"{fm['strike2']:.1f}°")
        st.metric("倾角", f"{fm['dip2']:.1f}°")
        st.metric("滑动角", f"{fm['rake2']:.1f}°")
    
    st.subheader("📊 拟合优度")
    misfit_percent = (1 - fm['misfit'] / fm['n_stations']) * 100
    st.metric("符合率", f"{misfit_percent:.1f}%", 
              f"{fm['n_stations'] - fm['misfit']}/{fm['n_stations']} 台站一致")
    
    st.subheader("🎨 Beach Ball 图")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_beach_ball(ax, fm['strike1'], fm['dip1'], fm['rake1'])
    
    polarity_data = fm['polarity_data']
    for i in range(len(polarity_data['stations'])):
        az = polarity_data['azimuths'][i]
        takeoff = polarity_data['takeoff_angles'][i]
        pol = polarity_data['polarities'][i]
        
        x, y = equal_area_project(az, takeoff)
        
        if pol > 0:
            ax.scatter(x, y, s=100, c='black', marker='o', edgecolors='black', zorder=10)
        else:
            ax.scatter(x, y, s=100, c='white', marker='o', edgecolors='black', zorder=10)
    
    ax.set_title('震源机制解 (P波初动)')
    ax.set_aspect('equal')
    ax.axis('off')
    
    st.pyplot(fig)
    plt.close(fig)
    
    st.subheader("📋 各台站极性数据")
    pol_df = pd.DataFrame({
        '台站': polarity_data['stations'],
        '方位角(°)': [f"{a:.1f}" for a in polarity_data['azimuths']],
        '出射角(°)': [f"{a:.1f}" for a in polarity_data['takeoff_angles']],
        '极性': ['压缩(+)' if p > 0 else '膨胀(-)' for p in polarity_data['polarities']]
    })
    st.dataframe(pol_df, use_container_width=True, hide_index=True)


def equal_area_project(azimuth, takeoff_angle):
    az_rad = np.radians(azimuth)
    takeoff_rad = np.radians(takeoff_angle)
    
    r = np.sqrt(2) * np.sin(takeoff_rad / 2)
    
    x = r * np.sin(az_rad)
    y = r * np.cos(az_rad)
    
    return x, y


def draw_beach_ball(ax, strike, dip, rake):
    import matplotlib.patches as patches
    from matplotlib.path import Path
    
    circle = patches.Circle((0, 0), np.sqrt(2)/2, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    
    plot_nodal_plane(ax, strike, dip, 'k-', linewidth=2)
    
    strike2, dip2, rake2 = compute_auxiliary_plane(strike, dip, rake)
    plot_nodal_plane(ax, strike2, dip2, 'k--', linewidth=1.5)
    
    fill_compressional(ax, strike, dip, rake)


def plot_nodal_plane(ax, strike, dip, *args, **kwargs):
    n_points = 100
    
    strike_rad = np.radians(strike)
    dip_rad = np.radians(dip)
    
    if dip < 89:
        plunge = np.arcsin(np.sin(dip_rad) * np.sin(strike_rad - strike_rad))
        
        thetas = np.linspace(-np.pi/2, np.pi/2, n_points)
        
        xs = []
        ys = []
        
        for theta in thetas:
            takeoff = np.arccos(np.sin(theta) * np.sin(dip_rad) + 
                                np.cos(theta) * np.cos(dip_rad) * np.cos(0))
            az = strike_rad + np.arcsin(np.cos(theta) * np.sin(0) / np.sin(takeoff))
            
            if not np.isnan(takeoff) and not np.isnan(az):
                x, y = equal_area_project(np.degrees(az), np.degrees(takeoff))
                xs.append(x)
                ys.append(y)
        
        ax.plot(xs, ys, *args, **kwargs)


def fill_compressional(ax, strike, dip, rake):
    n_pts = 500
    xs = []
    ys = []
    
    for az_deg in np.linspace(0, 360, n_pts):
        for takeoff_deg in np.linspace(0, 90, n_pts // 2):
            calc_pol = compute_p_for_nodal_plane(strike, dip, rake, takeoff_deg, az_deg)
            if calc_pol > 0:
                x, y = equal_area_project(az_deg, takeoff_deg)
                xs.append(x)
                ys.append(y)
    
    if xs:
        ax.scatter(xs, ys, s=1, c='black', alpha=0.1, marker='.')
