import numpy as np
from obspy import Trace, Stream, UTCDateTime
import os


def generate_sample_data(output_dir="sample_data"):
    os.makedirs(output_dir, exist_ok=True)
    
    stations = [
        {'name': 'STA01', 'lat': 39.9, 'lon': 116.3, 'elev': 50.0},
        {'name': 'STA02', 'lat': 39.8, 'lon': 116.5, 'elev': 80.0},
        {'name': 'STA03', 'lat': 40.0, 'lon': 116.7, 'elev': 120.0},
        {'name': 'STA04', 'lat': 39.7, 'lon': 116.1, 'elev': 60.0},
        {'name': 'STA05', 'lat': 40.1, 'lon': 116.2, 'elev': 100.0},
    ]
    
    epi_lat = 39.95
    epi_lon = 116.45
    depth_km = 15.0
    
    vp = 6.0
    vs = 3.5
    
    sampling_rate = 100.0
    duration = 120.0
    npts = int(sampling_rate * duration)
    
    origin_time = UTCDateTime('2024-01-15T08:30:00')
    
    import csv
    with open(os.path.join(output_dir, 'stations.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['station', 'longitude', 'latitude', 'elevation'])
        for sta in stations:
            writer.writerow([sta['name'], sta['lon'], sta['lat'], sta['elev']])
    
    st = Stream()
    
    for sta in stations:
        dist_deg_lat = sta['lat'] - epi_lat
        dist_deg_lon = sta['lon'] - epi_lon
        dist_km = np.sqrt((dist_deg_lat * 111)**2 + (dist_deg_lon * 111 * np.cos(np.radians(epi_lat)))**2)
        
        distance_3d = np.sqrt(dist_km**2 + depth_km**2)
        
        p_travel = distance_3d / vp
        s_travel = distance_3d / vs
        
        for comp in ['Z', 'N', 'E']:
            data = np.zeros(npts)
            
            noise = np.random.normal(0, 1e-8, npts)
            data += noise
            
            p_start_idx = int(p_travel * sampling_rate)
            s_start_idx = int(s_travel * sampling_rate)
            
            p_duration = int(2.0 * sampling_rate)
            s_duration = int(4.0 * sampling_rate)
            
            p_amp = 2e-6 / (dist_km + 10)
            s_amp = 5e-6 / (dist_km + 10)
            
            if p_start_idx + p_duration < npts:
                t_p = np.arange(p_duration) / sampling_rate
                if comp == 'Z':
                    p_wave = p_amp * np.sin(2 * np.pi * 5 * t_p) * np.exp(-t_p * 1.5)
                else:
                    p_wave = p_amp * 0.5 * np.sin(2 * np.pi * 5 * t_p) * np.exp(-t_p * 1.5)
                data[p_start_idx:p_start_idx + p_duration] += p_wave
            
            if s_start_idx + s_duration < npts:
                t_s = np.arange(s_duration) / sampling_rate
                if comp == 'Z':
                    s_wave = s_amp * 0.6 * np.sin(2 * np.pi * 3 * t_s) * np.exp(-t_s * 0.8)
                else:
                    s_wave = s_amp * np.sin(2 * np.pi * 3 * t_s) * np.exp(-t_s * 0.8)
                data[s_start_idx:s_start_idx + s_duration] += s_wave
            
            tr = Trace(data=data)
            tr.stats.network = 'XX'
            tr.stats.station = sta['name']
            tr.stats.channel = f'HH{comp}'
            tr.stats.sampling_rate = sampling_rate
            tr.stats.starttime = origin_time
            
            st.append(tr)
    
    mseed_path = os.path.join(output_dir, 'earthquake.mseed')
    sac_dir = os.path.join(output_dir, 'sac_files')
    os.makedirs(sac_dir, exist_ok=True)
    
    st.write(mseed_path, format='MSEED')
    
    for tr in st:
        sac_file = os.path.join(sac_dir, f"{tr.stats.station}_{tr.stats.channel[-1]}.SAC")
        tr.write(sac_file, format='SAC')
    
    print(f"示例数据已生成到: {output_dir}")
    print(f"  - MiniSEED: {mseed_path}")
    print(f"  - SAC文件: {sac_dir}/")
    print(f"  - 台站元数据: {os.path.join(output_dir, 'stations.csv')}")
    print(f"\n地震参数:")
    print(f"  - 震中: {epi_lat}°N, {epi_lon}°E")
    print(f"  - 深度: {depth_km} km")
    print(f"  - 发震时刻: {origin_time}")
    print(f"  - 台站数: {len(stations)}")


if __name__ == '__main__':
    generate_sample_data()
