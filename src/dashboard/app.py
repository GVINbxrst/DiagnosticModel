"""
Dashboard для диагностики асинхронных двигателей
Streamlit приложение с JWT авторизацией, визуализацией сигналов и отчетами
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import io
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import base64
import logging

# Импорты локальных модулей
from .utils import (
    SessionManager, DataCache, DataProcessor,
    ValidationUtils, FormatUtils, SecurityUtils, ConfigManager
)
from .components import (
    UIComponents, ChartComponents, FilterComponents,
    ExportComponents, NotificationComponents, ConfigComponents
)
from .pages import AdminPage, render_trends

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Конфигурация приложения
config = ConfigManager.load_dashboard_config()
API_BASE_URL = config.get('api_base_url', "http://api:8000")

st.set_page_config(
    page_title="DiagMod - Диагностика двигателей",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AuthManager:
    """Менеджер авторизации через JWT"""
    
    @staticmethod
    def login(username: str, password: str) -> Optional[Dict]:
        """Авторизация пользователя"""
        try:
            response = requests.post(
                f"{API_BASE_URL}/auth/login",
                data={"username": username, "password": password}
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"Ошибка авторизации: {e}")
            return None
    
    @staticmethod
    def verify_token(token: str) -> bool:
        """Проверка валидности токена"""
        try:
            response = requests.get(
                f"{API_BASE_URL}/auth/verify",
                headers={"Authorization": f"Bearer {token}"}
            )
            return response.status_code == 200
        except Exception:
            return False
    
    @staticmethod
    def get_user_info(token: str) -> Optional[Dict]:
        """Получение информации о пользователе"""
        try:
            response = requests.get(
                f"{API_BASE_URL}/auth/me",
                headers={"Authorization": f"Bearer {token}"}
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None

class DataManager:
    """Менеджер данных для работы с API"""
    
    def __init__(self, token: str):
        self.token = token
        self.headers = {"Authorization": f"Bearer {token}"}
    
    def get_equipment_list(self) -> List[Dict]:
        """Получение списка оборудования"""
        try:
            response = requests.get(
                f"{API_BASE_URL}/equipment",
                headers=self.headers
            )
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            logger.error(f"Ошибка получения списка оборудования: {e}")
            return []
    
    def get_equipment_files(self, equipment_id: int) -> List[Dict]:
        """Получение списка файлов для оборудования"""
        try:
            response = requests.get(
                f"{API_BASE_URL}/equipment/{equipment_id}/files",
                headers=self.headers
            )
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            logger.error(f"Ошибка получения файлов: {e}")
            return []
    
    def get_signal_data(self, raw_id: int) -> Optional[Dict]:
        """Получение данных сигнала"""
        try:
            response = requests.get(
                f"{API_BASE_URL}/signals/{raw_id}",
                headers=self.headers
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"Ошибка получения сигнала: {e}")
            return None
    
    def get_anomalies(self, equipment_id: int) -> List[Dict]:
        """Получение аномалий для оборудования"""
        try:
            response = requests.get(
                f"{API_BASE_URL}/anomalies/{equipment_id}",
                headers=self.headers
            )
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            logger.error(f"Ошибка получения аномалий: {e}")
            return []
    
    def get_features(self, raw_id: int) -> Optional[Dict]:
        """Получение признаков сигнала"""
        try:
            response = requests.get(
                f"{API_BASE_URL}/features/{raw_id}",
                headers=self.headers
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"Ошибка получения признаков: {e}")
            return None

class Visualizer:
    """Класс для создания визуализаций"""
    
    @staticmethod
    def plot_time_series(signal_data: Dict, title: str = "Токовые сигналы"):
        """Построение временных рядов"""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=('Фаза R', 'Фаза S', 'Фаза T'),
            vertical_spacing=0.05
        )
        
        # Создание временной оси
        sample_rate = signal_data.get('sample_rate_hz', 25600)
        samples_count = signal_data.get('samples_count', 0)
        time_axis = np.linspace(0, samples_count / sample_rate, samples_count)
        
        phases = ['phase_a', 'phase_b', 'phase_c']
        colors = ['red', 'green', 'blue']
        
        for i, (phase, color) in enumerate(zip(phases, colors)):
            if phase in signal_data and signal_data[phase]:
                phase_data = np.array(signal_data[phase])
                fig.add_trace(
                    go.Scatter(
                        x=time_axis[:len(phase_data)],
                        y=phase_data,
                        name=f'Фаза {phases[i][-1].upper()}',
                        line=dict(color=color, width=1),
                        showlegend=True
                    ),
                    row=i+1, col=1
                )
        
        fig.update_layout(
            title=title,
            height=800,
            showlegend=True
        )
        fig.update_xaxes(title_text="Время (с)", row=3, col=1)
        fig.update_yaxes(title_text="Ток (А)")
        
        return fig
    
    @staticmethod
    def plot_fft_spectrum(signal_data: Dict, features: Dict):
        """Построение FFT спектра"""
        if not features or 'fft_spectrum' not in features:
            return None
        
        fft_data = features['fft_spectrum']
        if not fft_data:
            return None
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=('FFT Фаза R', 'FFT Фаза S', 'FFT Фаза T'),
            vertical_spacing=0.05
        )
        
        phases = ['phase_a', 'phase_b', 'phase_c']
        colors = ['red', 'green', 'blue']
        
        for i, (phase, color) in enumerate(zip(phases, colors)):
            if phase in fft_data:
                spectrum = fft_data[phase]
                frequencies = spectrum.get('frequencies', [])
                magnitudes = spectrum.get('magnitudes', [])
                
                if frequencies and magnitudes:
                    fig.add_trace(
                        go.Scatter(
                            x=frequencies,
                            y=magnitudes,
                            name=f'FFT {phases[i][-1].upper()}',
                            line=dict(color=color, width=1)
                        ),
                        row=i+1, col=1
                    )
        
        fig.update_layout(
            title="Частотный спектр (FFT)",
            height=800
        )
        fig.update_xaxes(title_text="Частота (Гц)", row=3, col=1)
        fig.update_yaxes(title_text="Амплитуда")
        
        return fig
    
    @staticmethod
    def plot_rms_trend(anomalies: List[Dict]):
        """График тренда RMS и аномалий"""
        if not anomalies:
            return None
        
        # Извлечение данных RMS по времени
        rms_data = []
        for anomaly in anomalies:
            if 'rms_values' in anomaly:
                rms_data.extend(anomaly['rms_values'])
        
        if not rms_data:
            return None
        
        df = pd.DataFrame(rms_data)
        
        fig = go.Figure()
        
        # График RMS по фазам
        for phase in ['rms_a', 'rms_b', 'rms_c']:
            if phase in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'] if 'timestamp' in df.columns else range(len(df)),
                        y=df[phase],
                        name=f'RMS {phase[-1].upper()}',
                        mode='lines'
                    )
                )
        
        # Отметка аномалий
        anomaly_times = [a['detected_at'] for a in anomalies if a.get('is_anomaly')]
        if anomaly_times:
            for time in anomaly_times:
                fig.add_vline(
                    x=time,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Аномалия"
                )
        
        fig.update_layout(
            title="Тренд RMS и обнаруженные аномалии",
            xaxis_title="Время",
            yaxis_title="RMS (А)",
            height=500
        )
        
        return fig

class ReportGenerator:
    """Генератор PDF отчетов"""
    
    @staticmethod
    def generate_report(equipment_data: Dict, anomalies: List[Dict], 
                       signal_data: Dict, features: Dict) -> bytes:
        """Генерация PDF отчета"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        # Заголовок отчета
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            alignment=1  # По центру
        )
        
        story.append(Paragraph("Отчет диагностики асинхронного двигателя", title_style))
        story.append(Spacer(1, 12))
        
        # Информация об оборудовании
        equipment_info = [
            ['Параметр', 'Значение'],
            ['ID оборудования', str(equipment_data.get('id', 'N/A'))],
            ['Название', equipment_data.get('name', 'N/A')],
            ['Модель', equipment_data.get('model', 'N/A')],
            ['Дата анализа', datetime.now().strftime('%d.%m.%Y %H:%M')]
        ]
        
        equipment_table = Table(equipment_info)
        equipment_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(equipment_table)
        story.append(Spacer(1, 20))
        
        # Статистика признаков
        if features:
            story.append(Paragraph("Статистические характеристики", styles['Heading2']))
            
            features_info = []
            phases = ['a', 'b', 'c']
            
            for phase in phases:
                rms_val = features.get(f'rms_{phase}', 'N/A')
                crest_val = features.get(f'crest_{phase}', 'N/A')
                kurt_val = features.get(f'kurt_{phase}', 'N/A')
                skew_val = features.get(f'skew_{phase}', 'N/A')
                
                features_info.extend([
                    [f'Фаза {phase.upper()}', '', '', ''],
                    ['RMS', f'{rms_val:.3f}' if isinstance(rms_val, (int, float)) else str(rms_val), '', ''],
                    ['Crest Factor', f'{crest_val:.3f}' if isinstance(crest_val, (int, float)) else str(crest_val), '', ''],
                    ['Kurtosis', f'{kurt_val:.3f}' if isinstance(kurt_val, (int, float)) else str(kurt_val), '', ''],
                    ['Skewness', f'{skew_val:.3f}' if isinstance(skew_val, (int, float)) else str(skew_val), '', '']
                ])
            
            features_table = Table(features_info)
            story.append(features_table)
            story.append(Spacer(1, 20))
        
        # Результаты диагностики
        story.append(Paragraph("Результаты диагностики", styles['Heading2']))
        
        if anomalies:
            anomaly_count = len([a for a in anomalies if a.get('is_anomaly')])
            story.append(Paragraph(f"Обнаружено аномалий: {anomaly_count}", styles['Normal']))
            
            if anomaly_count > 0:
                anomaly_data = [['Время обнаружения', 'Тип', 'Вероятность', 'Серьезность']]
                for anomaly in anomalies:
                    if anomaly.get('is_anomaly'):
                        anomaly_data.append([
                            anomaly.get('detected_at', 'N/A'),
                            anomaly.get('defect_type', 'Неизвестно'),
                            f"{anomaly.get('probability', 0):.3f}",
                            anomaly.get('predicted_severity', 'N/A')
                        ])
                
                anomaly_table = Table(anomaly_data)
                anomaly_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(anomaly_table)
        else:
            story.append(Paragraph("Аномалий не обнаружено", styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Рекомендации
        story.append(Paragraph("Рекомендации", styles['Heading2']))
        if anomalies and any(a.get('is_anomaly') for a in anomalies):
            recommendations = [
                "• Провести детальную диагностику выявленных аномалий",
                "• Увеличить частоту мониторинга состояния двигателя",
                "• Рассмотреть возможность планового технического обслуживания",
                "• Проанализировать условия эксплуатации оборудования"
            ]
        else:
            recommendations = [
                "• Двигатель работает в нормальном режиме",
                "• Рекомендуется продолжить регулярный мониторинг",
                "• Следующая диагностика через установленный интервал"
            ]
        
        for rec in recommendations:
            story.append(Paragraph(rec, styles['Normal']))
        
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

def main():
    """Основная функция приложения"""
    
    # Заголовок приложения
    st.title("⚡ DiagMod - Система диагностики асинхронных двигателей")
    
    # Проверка авторизации
    if 'token' not in st.session_state or not st.session_state.get('token'):
        show_login()
        return
    
    # Проверка валидности токена
    if not AuthManager.verify_token(st.session_state.token):
        st.error("Сессия истекла. Пожалуйста, авторизуйтесь заново.")
        st.session_state.clear()
        st.rerun()
        return
    
    # Получение информации о пользователе
    user_info = AuthManager.get_user_info(st.session_state.token)
    if not user_info:
        st.error("Ошибка получения информации о пользователе")
        return
    
    # Боковая панель с информацией о пользователе
    with st.sidebar:
        st.write(f"👤 Пользователь: {user_info.get('username', 'N/A')}")
        st.write(f"🔒 Роль: {user_info.get('role', 'N/A')}")
        
        if st.button("Выйти"):
            st.session_state.clear()
            st.rerun()
        
        st.divider()
    
    # Основной интерфейс
    show_dashboard()

def show_login():
    """Форма авторизации"""
    st.subheader("🔐 Авторизация")
    
    with st.form("login_form"):
        username = st.text_input("Имя пользователя")
        password = st.text_input("Пароль", type="password")
        submitted = st.form_submit_button("Войти")
        
        if submitted:
            if username and password:
                auth_result = AuthManager.login(username, password)
                if auth_result and 'access_token' in auth_result:
                    st.session_state.token = auth_result['access_token']
                    st.session_state.user_info = auth_result.get('user_info', {})
                    st.success("Авторизация успешна!")
                    st.rerun()
                else:
                    st.error("Неверные учетные данные")
            else:
                st.error("Заполните все поля")

def show_dashboard():
    """Основной дашборд"""
    data_manager = DataManager(st.session_state.token)
    
    # Получение списка оборудования
    equipment_list = data_manager.get_equipment_list()
    
    if not equipment_list:
        st.warning("Нет доступного оборудования или ошибка загрузки данных")
        return
    
    # Выбор оборудования
    equipment_options = {eq['name']: eq for eq in equipment_list}
    selected_equipment_name = st.selectbox(
        "🔧 Выберите оборудование:",
        options=list(equipment_options.keys())
    )
    
    if not selected_equipment_name:
        return
    
    selected_equipment = equipment_options[selected_equipment_name]
    equipment_id = selected_equipment['id']
    
    # Информация об оборудовании
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ID оборудования", equipment_id)
    with col2:
        st.metric("Модель", selected_equipment.get('model', 'N/A'))
    with col3:
        st.metric("Статус", selected_equipment.get('status', 'N/A'))
    
    # Вкладки функциональности
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Обзор файлов", "📈 Анализ сигналов", "⚠️ Аномалии", "📄 Отчеты", "📈 Тренды"])
    
    with tab1:
        show_files_overview(data_manager, equipment_id)
    
    with tab2:
        show_signal_analysis(data_manager, equipment_id)
    
    with tab3:
        show_anomalies(data_manager, equipment_id)
    
    with tab4:
        show_reports(data_manager, equipment_id, selected_equipment)
    with tab5:
        render_trends(API_BASE_URL, st.session_state.token, str(equipment_id))

def show_files_overview(data_manager: DataManager, equipment_id: int):
    """Обзор файлов оборудования"""
    st.subheader("📊 Обзор файлов данных")
    
    files = data_manager.get_equipment_files(equipment_id)
    
    if not files:
        st.info("Файлы данных не найдены")
        return
    
    # Статистика файлов
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Всего файлов", len(files))
    with col2:
        total_samples = sum(f.get('samples_count', 0) for f in files)
        st.metric("Всего образцов", f"{total_samples:,}")
    with col3:
        avg_duration = np.mean([f.get('duration_seconds', 0) for f in files if f.get('duration_seconds')])
        st.metric("Средняя длительность", f"{avg_duration:.1f} сек")
    with col4:
        latest_file = max(files, key=lambda x: x.get('recorded_at', ''), default={})
        latest_date = latest_file.get('recorded_at', 'N/A')
        st.metric("Последний файл", latest_date[:10] if latest_date != 'N/A' else 'N/A')
    
    # Таблица файлов
    if files:
        df_files = pd.DataFrame(files)
        st.dataframe(
            df_files[['id', 'recorded_at', 'samples_count', 'sample_rate_hz']],
            use_container_width=True
        )

def show_signal_analysis(data_manager: DataManager, equipment_id: int):
    """Анализ сигналов"""
    st.subheader("📈 Анализ токовых сигналов")
    
    files = data_manager.get_equipment_files(equipment_id)
    
    if not files:
        st.info("Файлы для анализа не найдены")
        return
    
    # Выбор файла для анализа
    file_options = {f"Файл {f['id']} ({f['recorded_at'][:19]})": f['id'] for f in files}
    selected_file = st.selectbox("Выберите файл для анализа:", options=list(file_options.keys()))
    
    if not selected_file:
        return
    
    raw_id = file_options[selected_file]
    
    # Загрузка данных сигнала
    with st.spinner("Загрузка данных сигнала..."):
        signal_data = data_manager.get_signal_data(raw_id)
        features = data_manager.get_features(raw_id)
    
    if not signal_data:
        st.error("Ошибка загрузки данных сигнала")
        return
    
    # Визуализация временных рядов
    st.subheader("Временные ряды токов")
    fig_time = Visualizer.plot_time_series(signal_data)
    if fig_time:
        st.plotly_chart(fig_time, use_container_width=True)
    
    # Визуализация FFT спектра
    if features:
        st.subheader("Частотный анализ (FFT)")
        fig_fft = Visualizer.plot_fft_spectrum(signal_data, features)
        if fig_fft:
            st.plotly_chart(fig_fft, use_container_width=True)
        
        # Статистические характеристики
        st.subheader("Статистические характеристики")
        col1, col2, col3 = st.columns(3)
        
        phases = ['a', 'b', 'c']
        phase_names = ['R', 'S', 'T']
        
        for i, (phase, name) in enumerate(zip(phases, phase_names)):
            with [col1, col2, col3][i]:
                st.write(f"**Фаза {name}:**")
                rms_val = features.get(f'rms_{phase}')
                crest_val = features.get(f'crest_{phase}')
                kurt_val = features.get(f'kurt_{phase}')
                skew_val = features.get(f'skew_{phase}')
                
                if rms_val is not None:
                    st.metric("RMS", f"{rms_val:.3f}")
                if crest_val is not None:
                    st.metric("Crest Factor", f"{crest_val:.3f}")
                if kurt_val is not None:
                    st.metric("Kurtosis", f"{kurt_val:.3f}")
                if skew_val is not None:
                    st.metric("Skewness", f"{skew_val:.3f}")

def show_anomalies(data_manager: DataManager, equipment_id: int):
    """Отображение аномалий и прогнозов"""
    st.subheader("⚠️ Диагностика и прогнозирова��ие")

    # Загрузка данных аномалий
    with st.spinner("Загрузка данных диагностики..."):
        anomalies = data_manager.get_anomalies(equipment_id)
    
    if not anomalies:
        st.info("Данные диагностики не найдены")
        return
    
    # Общая статистика
    anomaly_count = len([a for a in anomalies if a.get('is_anomaly')])
    total_predictions = len(anomalies)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Всего прогнозов", total_predictions)
    with col2:
        st.metric("Обнаружено аномалий", anomaly_count, delta=f"{anomaly_count/total_predictions*100:.1f}%" if total_predictions > 0 else "0%")
    with col3:
        last_check = max([a.get('created_at', '') for a in anomalies], default='N/A')
        st.metric("Последняя проверка", last_check[:19] if last_check != 'N/A' else 'N/A')
    
    # График тренда RMS
    fig_rms = Visualizer.plot_rms_trend(anomalies)
    if fig_rms:
        st.plotly_chart(fig_rms, use_container_width=True)
    
    # Таблица аномалий
    if anomaly_count > 0:
        st.subheader("Обнаруженные аномалии")
        anomaly_data = []
        for anomaly in anomalies:
            if anomaly.get('is_anomaly'):
                anomaly_data.append({
                    'Время': anomaly.get('created_at', 'N/A')[:19],
                    'Тип дефекта': anomaly.get('defect_type', 'Неизвестно'),
                    'Вероятность': f"{anomaly.get('probability', 0):.3f}",
                    'Серьезность': anomaly.get('predicted_severity', 'N/A'),
                    'Модель': anomaly.get('model_name', 'N/A')
                })
        
        if anomaly_data:
            df_anomalies = pd.DataFrame(anomaly_data)
            st.dataframe(df_anomalies, use_container_width=True)
    else:
        st.success("🟢 Аномалий не обнаружено. Оборудование работает в нормальном режиме.")

def show_reports(data_manager: DataManager, equipment_id: int, equipment_data: Dict):
    """Генерация отчетов"""
    st.subheader("📄 Генерация отчетов")
    
    # Выбор параметров отчета
    col1, col2 = st.columns(2)
    
    with col1:
        report_type = st.selectbox(
            "Тип отчета:",
            ["Полный диагностический отчет", "Отчет по аномалиям", "Сводка по оборудованию"]
        )
    
    with col2:
        include_charts = st.checkbox("Включить графики", value=True)
    
    # Получение данных для отчета
    if st.button("Сгенерировать отчет", type="primary"):
        with st.spinner("Генерация отчета..."):
            try:
                # Загрузка всех необходимых данных
                anomalies = data_manager.get_anomalies(equipment_id)
                files = data_manager.get_equipment_files(equipment_id)
                
                # Получение последнего файла для анализа
                signal_data = {}
                features = {}
                if files:
                    latest_file = max(files, key=lambda x: x.get('recorded_at', ''), default={})
                    if latest_file:
                        signal_data = data_manager.get_signal_data(latest_file['id']) or {}
                        features = data_manager.get_features(latest_file['id']) or {}
                
                # Генерация PDF отчета
                pdf_bytes = ReportGenerator.generate_report(
                    equipment_data, anomalies, signal_data, features
                )
                
                # Создание ссылки для скачивания
                b64_pdf = base64.b64encode(pdf_bytes).decode()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"diagmod_report_{equipment_id}_{timestamp}.pdf"
                
                href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{filename}">📥 Скачать отчет</a>'
                st.markdown(href, unsafe_allow_html=True)
                
                st.success("Отчет успешно сгенерирован!")
                
            except Exception as e:
                st.error(f"Ошибка генерации отчета: {e}")
                logger.error(f"Ошибка генерации отчета: {e}")

if __name__ == "__main__":
    main()
