import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import UTCDateTime
from geopy.distance import geodesic
import warnings
warnings.filterwarnings('ignore')


def location_page():
    st.title("📍 震源定位")
    st.markdown("---")

    if not st.session_state.picks:
        st.warning("请先进行震相拾取")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("速度模型参数")
        
        n_layers = st.number_input("层数", value=3, min_value=1, max_value=10)
        
        layers = []
        default_thickness = [10, 20, 0]
        default_vp = [5.5, 6.5, 8.0]
        default_vs = [3.2, 3.8, 4.5]
        
        for i in range(int(n_layers)):
            st.markdown(f"**第 {i+1} 层**")
            c1, c2, c3 = st.columns(3)
            with c1:
                thick = st.number_input(f"厚度(km) {i+1}", value=float(default_thickness[i]) if i < len(default_thickness) else 10.0, key=f"thick_{i}")
            with c2:
                vp = st.number_input(f"Vp(km/s) {i+1}", value=float(default_vp[i]) if i < len(default_vp) else 6.0, key=f"vp_{i}")
            with c3:
                vs = st.number_input(f"Vs(km/s) {i+1}", value=float(default_vs[i]) if i < len(default_vs) else 3.5, key=f"vs_{i}")
            layers.append({'thickness': thick, 'vp': vp, 'vs': vs})
        
        st.subheader("定位参数")
        max_iter = st.number_input("最大迭代次数", value=50, min_value=5)
        rms_threshold = st.number_input("RMS阈值 (s)", value=0.1, min_value=0.01, step=0.01)
        use_p = st.checkbox("使用P波", value=True)
        use_s = st.checkbox("使用S波", value=True)
        
        if st.button("🔍 执行定位", type="primary", use_container_width=True):
            with st.spinner("正在执行震源定位..."):
                result = locate_earthquake(layers, max_iter, rms_threshold, use_p, use_s)
                if result:
                    st.session_state.location_result = result
                    st.success("定位完成！")
                else:
                    st.error("定位失败，请检查输入数据")

    with col2:
        if st.session_state.location_result:
            display_location_result()
        else:
            st.info("👈 配置参数后点击执行定位")


def get_station_coords(station_name):
    if st.session_state.station_metadata is not None:
        meta = st.session_state.station_metadata
        row = meta[meta['station'] == station_name]
        if not row.empty:
            return float(row.iloc[0]['latitude']), float(row.iloc[0]['longitude'])
    
    if st.session_state.waveforms:
        for tr in st.session_state.waveforms:
            if tr.stats.station == station_name:
                if hasattr(tr.stats, 'sac'):
                    sac = tr.stats.sac
                    if hasattr(sac, 'stla') and hasattr(sac, 'stlo'):
                        return float(sac.stla), float(sac.stlo)
    
    return None, None


def latlon_to_xy(lat0, lon0, lat, lon):
    x = (lon - lon0) * 111.32 * np.cos(np.radians(lat0))
    y = (lat - lat0) * 111.32
    return x, y


def xy_to_latlon(lat0, lon0, x, y):
    lat = lat0 + y / 111.32
    lon = lon0 + x / (111.32 * np.cos(np.radians(lat0)))
    return lat, lon


def travel_time_1d(layers, distance_km, depth_km, phase='P'):
    velocities = []
    depths = []
    current_depth = 0
    
    for i, layer in enumerate(layers):
        if i == len(layers) - 1:
            thickness = 1000
        else:
            thickness = layer['thickness']
        vel = layer['vp'] if phase == 'P' else layer['vs']
        velocities.append(vel)
        depths.append(current_depth)
        current_depth += thickness
    
    source_layer = 0
    for i in range(len(depths)):
        if i < len(depths) - 1 and depth_km >= depths[i] and depth_km < depths[i+1]:
            source_layer = i
            break
        if i == len(depths) - 1:
            source_layer = i
    
    direct_time = np.sqrt(distance_km**2 + depth_km**2) / velocities[source_layer]
    
    refraction_times = []
    for i in range(source_layer + 1, len(velocities)):
        if velocities[i] > velocities[source_layer]:
            v1 = velocities[source_layer]
            v2 = velocities[i]
            h = depths[i] - depth_km if i > source_layer else 0
            if i == source_layer + 1:
                h = depths[i] - depth_km if i < len(depths) else depths[-1]
            
            sin_i = v1 / v2
            cos_i = np.sqrt(1 - sin_i**2)
            if cos_i > 0:
                x_crit = 2 * h * sin_i / cos_i
                if distance_km >= x_crit:
                    t = distance_km / v2 + 2 * h * cos_i / v1
                    refraction_times.append(t)
    
    all_times = [direct_time]
    if refraction_times:
        all_times.extend(refraction_times)
    
    return min(all_times)


def locate_earthquake(layers, max_iter, rms_threshold, use_p, use_s):
    picks = st.session_state.picks
    
    stations = []
    p_times = []
    s_times = []
    lats = []
    lons = []
    
    for sta, p in picks.items():
        lat, lon = get_station_coords(sta)
        if lat is None or lon is None:
            continue
        
        if use_p and 'P' in p and p['P']:
            stations.append(sta)
            p_times.append(p['P'])
            s_times.append(p.get('S', None))
            lats.append(lat)
            lons.append(lon)
        elif use_s and 'S' in p and p['S']:
            stations.append(sta)
            p_times.append(p.get('P', None))
            s_times.append(p['S'])
            lats.append(lat)
            lons.append(lon)
    
    if len(stations) < 3:
        st.error("至少需要3个台站的有效数据")
        return None
    
    lat0 = np.mean(lats)
    lon0 = np.mean(lons)
    
    x_stations = []
    y_stations = []
    for lat, lon in zip(lats, lons):
        x, y = latlon_to_xy(lat0, lon0, lat, lon)
        x_stations.append(x)
        y_stations.append(y)
    
    x_stations = np.array(x_stations)
    y_stations = np.array(y_stations)
    
    x0 = np.mean(x_stations)
    y0 = np.mean(y_stations)
    z0 = 10.0
    
    ref_time = None
    for t in p_times:
        if t is not None:
            ref_time = t
            break
    for t in s_times:
        if t is not None and (ref_time is None or t < ref_time):
            ref_time = t
    
    t0_sec = 0.0
    if ref_time:
        dist0 = np.sqrt((x_stations[0] - x0)**2 + (y_stations[0] - y0)**2 + z0**2)
        t_first = p_times[0] if p_times[0] else s_times[0]
        t0_sec = (t_first - ref_time) - dist0 / layers[0]['vp']
    
    params = np.array([x0, y0, z0, t0_sec])
    
    rms_history = []
    
    for iteration in range(int(max_iter)):
        residuals = []
        jacobian = []
        
        for i in range(len(stations)):
            dx = x_stations[i] - params[0]
            dy = y_stations[i] - params[1]
            dist_km = np.sqrt(dx**2 + dy**2)
            depth = params[2]
            
            if use_p and p_times[i] is not None:
                tt = travel_time_1d(layers, dist_km, depth, 'P')
                obs_t = (p_times[i] - ref_time)
                res = obs_t - (params[3] + tt)
                residuals.append(res)
                
                dist_3d = np.sqrt(dist_km**2 + depth**2)
                vp = layers[0]['vp']
                dtdx = -dx / (dist_3d * vp) if dist_3d > 0 else 0
                dtdy = -dy / (dist_3d * vp) if dist_3d > 0 else 0
                dtdz = depth / (dist_3d * vp) if dist_3d > 0 else 0
                dtdt = 1.0
                jacobian.append([dtdx, dtdy, dtdz, dtdt])
            
            if use_s and s_times[i] is not None:
                tt = travel_time_1d(layers, dist_km, depth, 'S')
                obs_t = (s_times[i] - ref_time)
                res = obs_t - (params[3] + tt)
                residuals.append(res)
                
                dist_3d = np.sqrt(dist_km**2 + depth**2)
                vs = layers[0]['vs']
                dtdx = -dx / (dist_3d * vs) if dist_3d > 0 else 0
                dtdy = -dy / (dist_3d * vs) if dist_3d > 0 else 0
                dtdz = depth / (dist_3d * vs) if dist_3d > 0 else 0
                dtdt = 1.0
                jacobian.append([dtdx, dtdy, dtdz, dtdt])
        
        residuals = np.array(residuals)
        jacobian = np.array(jacobian)
        
        rms = np.sqrt(np.mean(residuals**2))
        rms_history.append(rms)
        
        if rms < rms_threshold:
            break
        
        try:
            jtj = jacobian.T @ jacobian
            jtr = jacobian.T @ residuals
            delta = np.linalg.solve(jtj, jtr)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(jacobian, residuals, rcond=None)[0]
        
        params = params + delta
        
        if params[2] < 0:
            params[2] = 0.1
    
    epi_lat, epi_lon = xy_to_latlon(lat0, lon0, params[0], params[1])
    origin_time = ref_time + params[3]
    
    weights = []
    for i in range(len(stations)):
        dx = x_stations[i] - params[0]
        dy = y_stations[i] - params[1]
        dist = np.sqrt(dx**2 + dy**2 + params[2]**2)
        weights.append(1.0 / (dist + 1))
    
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    return {
        'latitude': epi_lat,
        'longitude': epi_lon,
        'depth_km': params[2],
        'origin_time': origin_time,
        'rms': rms_history[-1],
        'iterations': len(rms_history),
        'rms_history': rms_history,
        'stations': stations,
        'station_lats': lats,
        'station_lons': lons,
        'weights': weights,
        'x_stations': x_stations,
        'y_stations': y_stations,
        'x_epi': params[0],
        'y_epi': params[1],
        'ref_time': ref_time,
        'p_times': p_times,
        's_times': s_times,
        'layers': layers
    }


def display_location_result():
    result = st.session_state.location_result
    
    st.subheader("🎯 定位结果")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("震中纬度", f"{result['latitude']:.4f}°")
    with col_b:
        st.metric("震中经度", f"{result['longitude']:.4f}°")
    with col_c:
        st.metric("震源深度", f"{result['depth_km']:.2f} km")
    
    col_d, col_e, col_f = st.columns(3)
    with col_d:
        st.metric("发震时刻", str(result['origin_time']))
    with col_e:
        st.metric("RMS残差", f"{result['rms']:.3f} s")
    with col_f:
        st.metric("迭代次数", f"{result['iterations']}")
    
    st.subheader("🗺️ 震中位置")
    
    import plotly.graph_objects as go
    
    lats = result['station_lats'] + [result['latitude']]
    lons = result['station_lons'] + [result['longitude']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scattermap(
        lat=result['station_lats'],
        lon=result['station_lons'],
        mode='markers+text',
        marker=dict(size=10, color='red', opacity=0.8),
        text=result['stations'],
        textposition="top center",
        name='台站'
    ))
    
    fig.add_trace(go.Scattermap(
        lat=[result['latitude']],
        lon=[result['longitude']],
        mode='markers',
        marker=dict(size=15, color='gold', symbol='star', opacity=1.0),
        name='震中'
    ))
    
    fig.update_layout(
        map=dict(
            style="open-street-map",
            zoom=6,
            center=dict(lat=result['latitude'], lon=result['longitude'])
        ),
        showlegend=True,
        height=400,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📉 残差收敛曲线")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, len(result['rms_history']) + 1), result['rms_history'], 'b-o')
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('RMS残差 (s)')
        ax.set_title('定位收敛过程')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.subheader("📊 台站权重分布")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(result['stations'], result['weights'], color='steelblue')
        ax.set_xlabel('台站')
        ax.set_ylabel('权重')
        ax.set_title('各台站权重')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    st.subheader("⏱️ 走时曲线对比")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    distances = []
    obs_p = []
    obs_s = []
    theo_p = []
    theo_s = []
    
    for i, sta in enumerate(result['stations']):
        dx = result['x_stations'][i] - result['x_epi']
        dy = result['y_stations'][i] - result['y_epi']
        dist = np.sqrt(dx**2 + dy**2)
        distances.append(dist)
        
        if result['p_times'][i] is not None:
            obs_t = (result['p_times'][i] - result['ref_time'])
            obs_p.append((dist, obs_t))
            tt = travel_time_1d(result['layers'], dist, result['depth_km'], 'P')
            theo_p.append((dist, result['origin_time'] - result['ref_time'] + tt))
        
        if result['s_times'][i] is not None:
            obs_t = (result['s_times'][i] - result['ref_time'])
            obs_s.append((dist, obs_t))
            tt = travel_time_1d(result['layers'], dist, result['depth_km'], 'S')
            theo_s.append((dist, result['origin_time'] - result['ref_time'] + tt))
    
    if obs_p:
        obs_p.sort()
        dists_p = [d for d, t in obs_p]
        times_p = [t for d, t in obs_p]
        ax.scatter(dists_p, times_p, c='red', marker='o', label='P波观测', zorder=5)
    
    if obs_s:
        obs_s.sort()
        dists_s = [d for d, t in obs_s]
        times_s = [t for d, t in obs_s]
        ax.scatter(dists_s, times_s, c='blue', marker='s', label='S波观测', zorder=5)
    
    if theo_p:
        theo_p.sort()
        ax.plot([d for d, t in theo_p], [t for d, t in theo_p], 'r--', label='P波理论')
    
    if theo_s:
        theo_s.sort()
        ax.plot([d for d, t in theo_s], [t for d, t in theo_s], 'b--', label='S波理论')
    
    ax.set_xlabel('震中距 (km)')
    ax.set_ylabel('走时 (s)')
    ax.set_title('观测走时 vs 理论走时')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)
