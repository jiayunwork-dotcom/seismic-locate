import streamlit as st
import os
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
import warnings
warnings.filterwarnings('ignore')


def report_page():
    st.title("📄 报告输出")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("报告内容")
        
        include_location = st.checkbox("震源定位结果", value=True)
        include_magnitude = st.checkbox("震级计算结果", value=True)
        include_spectrum = st.checkbox("频谱分析结果", value=True)
        include_focal = st.checkbox("震源机制解", value=True)
        include_waveforms = st.checkbox("波形图", value=True)
        include_stations = st.checkbox("台站分布", value=True)
        
        report_title = st.text_input("报告标题", "地震分析报告")
        author = st.text_input("分析人员", "")
        notes = st.text_area("备注", "")
        
        if st.button("📑 生成PDF报告", type="primary", use_container_width=True):
            with st.spinner("正在生成PDF报告..."):
                pdf_path = generate_pdf_report(
                    report_title, author, notes,
                    include_location, include_magnitude,
                    include_spectrum, include_focal,
                    include_waveforms, include_stations
                )
                if pdf_path:
                    st.success("报告生成成功！")
                    with open(pdf_path, 'rb') as f:
                        st.download_button(
                            "⬇️ 下载PDF报告",
                            f,
                            file_name="seismic_report.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )

    with col2:
        st.subheader("📋 报告预览")
        st.info("PDF报告将包含以下内容：")
        
        items = []
        if include_location:
            items.append("✅ 震源定位结果（震中位置、深度、发震时刻、残差）")
        if include_magnitude:
            items.append("✅ 震级计算结果（ML、Mw、各台站详情）")
        if include_spectrum:
            items.append("✅ 频谱分析结果（振幅谱、拐角频率、应力降）")
        if include_focal:
            items.append("✅ 震源机制解（节面参数、Beach Ball图）")
        if include_waveforms:
            items.append("✅ 波形图（各台站波形与震相拾取标记）")
        if include_stations:
            items.append("✅ 台站分布图")
        
        for item in items:
            st.write(item)


def generate_pdf_report(title, author, notes, include_location, include_magnitude,
                        include_spectrum, include_focal, include_waveforms, include_stations):
    try:
        tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        tmp_pdf.close()
        
        doc = SimpleDocTemplate(tmp_pdf.name, pagesize=A4,
                                rightMargin=2*cm, leftMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)
        
        styles = getSampleStyleSheet()
        story = []
        
        title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=20, spaceAfter=20)
        story.append(Paragraph(title, title_style))
        
        if author:
            story.append(Paragraph(f"<b>分析人员：</b>{author}", styles['Normal']))
        story.append(Paragraph(f"<b>生成时间：</b>{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        
        if notes:
            story.append(Spacer(1, 0.5*cm))
            story.append(Paragraph(f"<b>备注：</b>{notes}", styles['Normal']))
        
        story.append(Spacer(1, 0.5*cm))
        
        if include_stations and st.session_state.waveforms is not None:
            story.append(Paragraph("1. 台站分布", styles['Heading2']))
            story.append(Spacer(1, 0.3*cm))
            
            img_path = save_station_map_image()
            if img_path:
                img = Image(img_path, width=15*cm, height=10*cm)
                story.append(img)
                story.append(Spacer(1, 0.5*cm))
        
        if include_waveforms and st.session_state.waveforms is not None:
            story.append(Paragraph("2. 波形数据", styles['Heading2']))
            story.append(Spacer(1, 0.3*cm))
            
            img_path = save_waveform_image()
            if img_path:
                img = Image(img_path, width=16*cm, height=10*cm)
                story.append(img)
                story.append(Spacer(1, 0.5*cm))
        
        if include_location and st.session_state.location_result:
            story.append(Paragraph("3. 震源定位结果", styles['Heading2']))
            story.append(Spacer(1, 0.3*cm))
            
            loc = st.session_state.location_result
            
            loc_data = [
                ['参数', '数值'],
                ['震中纬度', f"{loc['latitude']:.4f}°"],
                ['震中经度', f"{loc['longitude']:.4f}°"],
                ['震源深度', f"{loc['depth_km']:.2f} km"],
                ['发震时刻', str(loc['origin_time'])],
                ['RMS残差', f"{loc['rms']:.3f} s"],
                ['迭代次数', f"{loc['iterations']}"],
                ['使用台站数', f"{len(loc['stations'])}"]
            ]
            
            t = Table(loc_data, colWidths=[5*cm, 10*cm])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(t)
            
            story.append(Spacer(1, 0.5*cm))
            
            img_path = save_location_image()
            if img_path:
                img = Image(img_path, width=16*cm, height=8*cm)
                story.append(img)
            
            story.append(PageBreak())
        
        if include_magnitude and st.session_state.magnitude_result:
            story.append(Paragraph("4. 震级计算结果", styles['Heading2']))
            story.append(Spacer(1, 0.3*cm))
            
            mag = st.session_state.magnitude_result
            
            if 'ML' in mag and mag['ML']:
                story.append(Paragraph("4.1 ML 本地震级", styles['Heading3']))
                ml = mag['ML']
                
                ml_summary = [
                    ['参数', '数值'],
                    ['ML均值', f"{ml['ML_mean']:.2f}"],
                    ['ML标准差', f"{ml['ML_std']:.2f}"],
                    ['台站数', f"{ml['n_stations']}"]
                ]
                
                t = Table(ml_summary, colWidths=[5*cm, 10*cm])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(t)
                story.append(Spacer(1, 0.3*cm))
            
            if 'Mw' in mag and mag['Mw']:
                story.append(Paragraph("4.2 Mw 矩震级", styles['Heading3']))
                mw = mag['Mw']
                
                mw_summary = [
                    ['参数', '数值'],
                    ['Mw', f"{mw['Mw']:.2f}"],
                    ['标量地震矩 M₀', f"{mw['M0']:.2e} N·m"],
                    ['台站数', f"{mw['n_stations']}"]
                ]
                
                t = Table(mw_summary, colWidths=[5*cm, 10*cm])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(t)
            
            story.append(Spacer(1, 0.5*cm))
            
            story.append(PageBreak())
        
        if include_spectrum and hasattr(st.session_state, 'spec_results') and st.session_state.spec_results:
            story.append(Paragraph("5. 频谱分析结果", styles['Heading2']))
            story.append(Spacer(1, 0.3*cm))
            
            img_path = save_spectrum_image()
            if img_path:
                img = Image(img_path, width=16*cm, height=10*cm)
                story.append(img)
            
            story.append(PageBreak())
        
        if include_focal and st.session_state.focal_mechanism:
            story.append(Paragraph("6. 震源机制解", styles['Heading2']))
            story.append(Spacer(1, 0.3*cm))
            
            fm = st.session_state.focal_mechanism
            
            fm_data = [
                ['参数', '节面 I', '节面 II'],
                ['走向', f"{fm['strike1']:.1f}°", f"{fm['strike2']:.1f}°"],
                ['倾角', f"{fm['dip1']:.1f}°", f"{fm['dip2']:.1f}°"],
                ['滑动角', f"{fm['rake1']:.1f}°", f"{fm['rake2']:.1f}°"]
            ]
            
            t = Table(fm_data, colWidths=[4*cm, 5.5*cm, 5.5*cm])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(t)
            
            story.append(Spacer(1, 0.3*cm))
            
            misfit_pct = (1 - fm['misfit'] / fm['n_stations']) * 100
            story.append(Paragraph(f"拟合符合率: {misfit_pct:.1f}% ({fm['n_stations'] - fm['misfit']}/{fm['n_stations']} 台站)", styles['Normal']))
            
            story.append(Spacer(1, 0.3*cm))
            
            img_path = save_beachball_image()
            if img_path:
                img = Image(img_path, width=10*cm, height=10*cm)
                story.append(img)
        
        doc.build(story)
        
        return tmp_pdf.name
    
    except Exception as e:
        st.error(f"生成PDF失败: {e}")
        return None


def save_station_map_image():
    try:
        tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        tmp_img.close()
        
        lats = []
        lons = []
        names = []
        
        if st.session_state.station_metadata is not None:
            meta = st.session_state.station_metadata
            lats = meta['latitude'].tolist()
            lons = meta['longitude'].tolist()
            names = meta['station'].tolist()
        elif st.session_state.location_result:
            lats = st.session_state.location_result['station_lats']
            lons = st.session_state.location_result['station_lons']
            names = st.session_state.location_result['stations']
        
        if not lats:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(lons, lats, c='red', s=100, marker='^', edgecolors='black', zorder=5)
        
        for i, name in enumerate(names):
            ax.annotate(name, (lons[i], lats[i]), fontsize=8, ha='left', va='bottom')
        
        if st.session_state.location_result:
            loc = st.session_state.location_result
            ax.scatter(loc['longitude'], loc['latitude'], c='gold', s=200, marker='*', 
                       edgecolors='black', zorder=10, label='震中')
        
        ax.set_xlabel('经度 (°)')
        ax.set_ylabel('纬度 (°)')
        ax.set_title('台站分布图')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(tmp_img.name, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return tmp_img.name
    except:
        return None


def save_waveform_image():
    try:
        tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        tmp_img.close()
        
        waveforms = st.session_state.processed_waveforms or st.session_state.waveforms
        picks = st.session_state.picks if st.session_state.picks else {}
        
        stations = sorted(set(tr.stats.station for tr in waveforms))
        
        if not stations:
            return None
        
        n_stations = min(len(stations), 6)
        fig, axes = plt.subplots(n_stations, 1, figsize=(12, 2*n_stations), sharex=True)
        
        if n_stations == 1:
            axes = [axes]
        
        for i, sta in enumerate(stations[:n_stations]):
            z_tr = None
            for tr in waveforms:
                if tr.stats.station == sta and tr.stats.channel[-1] == 'Z':
                    z_tr = tr
                    break
            
            if z_tr is not None:
                times = z_tr.times()
                axes[i].plot(times, z_tr.data, 'k-', linewidth=0.5)
                axes[i].set_ylabel(sta, rotation=0, labelpad=20, ha='right')
                
                if sta in picks and 'P' in picks[sta]:
                    p_time = picks[sta]['P']
                    p_offset = p_time - z_tr.stats.starttime
                    axes[i].axvline(p_offset, color='r', linestyle='--', linewidth=1)
                
                axes[i].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('时间 (s)')
        fig.suptitle('波形记录', y=1.02)
        
        plt.tight_layout()
        fig.savefig(tmp_img.name, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return tmp_img.name
    except:
        return None


def save_location_image():
    try:
        tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        tmp_img.close()
        
        loc = st.session_state.location_result
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        rms_hist = loc['rms_history']
        axes[0].plot(range(1, len(rms_hist) + 1), rms_hist, 'b-o')
        axes[0].set_xlabel('迭代次数')
        axes[0].set_ylabel('RMS残差 (s)')
        axes[0].set_title('残差收敛曲线')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].bar(loc['stations'], loc['weights'], color='steelblue')
        axes[1].set_xlabel('台站')
        axes[1].set_ylabel('权重')
        axes[1].set_title('各台站权重')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        fig.savefig(tmp_img.name, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return tmp_img.name
    except:
        return None


def save_spectrum_image():
    try:
        tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        tmp_img.close()
        
        results = st.session_state.spec_results
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
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
        
        plt.tight_layout()
        fig.savefig(tmp_img.name, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return tmp_img.name
    except:
        return None


def save_beachball_image():
    try:
        tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        tmp_img.close()
        
        fm = st.session_state.focal_mechanism
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        from modules.focal_mechanism import equal_area_project, compute_p_for_nodal_plane, compute_auxiliary_plane
        
        import matplotlib.patches as patches
        
        circle = patches.Circle((0, 0), np.sqrt(2)/2, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        
        n_pts = 300
        xs_c = []
        ys_c = []
        xs_d = []
        ys_d = []
        
        for az_deg in np.linspace(0, 360, n_pts):
            for takeoff_deg in np.linspace(0, 90, n_pts // 3):
                calc_pol = compute_p_for_nodal_plane(fm['strike1'], fm['dip1'], fm['rake1'], 
                                                    takeoff_deg, az_deg)
                x, y = equal_area_project(az_deg, takeoff_deg)
                if calc_pol > 0:
                    xs_c.append(x)
                    ys_c.append(y)
                else:
                    xs_d.append(x)
                    ys_d.append(y)
        
        ax.scatter(xs_c, ys_c, s=0.5, c='black', alpha=0.3, marker='.')
        
        def draw_np(strike, dip, style='-', width=2):
            thetas = np.linspace(0, 2*np.pi, 200)
            strike_rad = np.radians(strike)
            dip_rad = np.radians(dip)
            
            if dip < 89:
                xs = []
                ys = []
                for theta in np.linspace(-np.pi/2, np.pi/2, 100):
                    takeoff_rad = np.arccos(np.sin(theta) * np.sin(dip_rad) + 
                                            np.cos(theta) * np.cos(dip_rad) * np.cos(0))
                    az = strike_rad + np.arcsin(np.cos(theta) * np.sin(0) / np.sin(takeoff_rad))
                    
                    if not np.isnan(takeoff_rad) and not np.isnan(az):
                        x, y = equal_area_project(np.degrees(az), np.degrees(takeoff_rad))
                        xs.append(x)
                        ys.append(y)
                ax.plot(xs, ys, 'k' + style, linewidth=width)
        
        draw_np(fm['strike1'], fm['dip1'], '-', 2)
        draw_np(fm['strike2'], fm['dip2'], '--', 1.5)
        
        polarity_data = fm['polarity_data']
        for i in range(len(polarity_data['stations'])):
            az = polarity_data['azimuths'][i]
            takeoff = polarity_data['takeoff_angles'][i]
            pol = polarity_data['polarities'][i]
            
            x, y = equal_area_project(az, takeoff)
            
            if pol > 0:
                ax.scatter(x, y, s=80, c='black', marker='o', edgecolors='black', zorder=10)
            else:
                ax.scatter(x, y, s=80, c='white', marker='o', edgecolors='black', zorder=10)
        
        ax.set_title('震源机制解 (P波初动)')
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.tight_layout()
        fig.savefig(tmp_img.name, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return tmp_img.name
    except Exception as e:
        print(f"Beach ball error: {e}")
        return None
