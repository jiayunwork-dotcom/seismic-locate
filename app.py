import streamlit as st
from streamlit_option_menu import option_menu
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="地震波形数据处理与震源定位系统",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'waveforms' not in st.session_state:
    st.session_state.waveforms = None
if 'station_metadata' not in st.session_state:
    st.session_state.station_metadata = None
if 'processed_waveforms' not in st.session_state:
    st.session_state.processed_waveforms = None
if 'picks' not in st.session_state:
    st.session_state.picks = {}
if 'location_result' not in st.session_state:
    st.session_state.location_result = None
if 'magnitude_result' not in st.session_state:
    st.session_state.magnitude_result = None
if 'focal_mechanism' not in st.session_state:
    st.session_state.focal_mechanism = None

with st.sidebar:
    selected = option_menu(
        menu_title="地震数据分析系统",
        options=["数据导入", "波形预处理", "震相自动拾取", "震源定位", "震级计算", "频谱分析", "震源机制解", "报告输出"],
        icons=["cloud-upload", "sliders", "cursor", "pin-map", "graph-up", "bar-chart", "circle", "file-pdf"],
        menu_icon="activity",
        default_index=0,
    )

if selected == "数据导入":
    from modules.data_import import data_import_page
    data_import_page()
elif selected == "波形预处理":
    from modules.preprocessing import preprocessing_page
    preprocessing_page()
elif selected == "震相自动拾取":
    from modules.phase_picking import phase_picking_page
    phase_picking_page()
elif selected == "震源定位":
    from modules.location import location_page
    location_page()
elif selected == "震级计算":
    from modules.magnitude import magnitude_page
    magnitude_page()
elif selected == "频谱分析":
    from modules.spectrum import spectrum_page
    spectrum_page()
elif selected == "震源机制解":
    from modules.focal_mechanism import focal_mechanism_page
    focal_mechanism_page()
elif selected == "报告输出":
    from modules.report import report_page
    report_page()
