"""
Компоненты пользовательского интерфейса для Dashboard
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta

class UIComponents:
    """Переиспользуемые UI компоненты"""

    @staticmethod
    def create_metric_card(title: str, value: str, delta: Optional[str] = None,
                          delta_color: str = "normal"):
        """Создание карточки метрики"""
        col = st.columns(1)[0]
        with col:
            st.metric(
                label=title,
                value=value,
                delta=delta,
                delta_color=delta_color
            )

    @staticmethod
    def create_status_indicator(status: str, label: str = "Статус"):
        """Создание индикатора статуса"""
        status_colors = {
            "normal": "🟢",
            "warning": "🟡",
            "critical": "🔴",
            "unknown": "⚪"
        }

        icon = status_colors.get(status.lower(), "⚪")
        st.write(f"{label}: {icon} {status.title()}")

    @staticmethod
    def create_data_table(data: List[Dict], columns: List[str],
                         title: Optional[str] = None):
        """Создание таблицы данных"""
        if title:
            st.subheader(title)

        if not data:
            st.info("Нет данных для отображения")
            return

        df = pd.DataFrame(data)
        if columns:
            df = df[columns] if all(col in df.columns for col in columns) else df

        st.dataframe(df, use_container_width=True)

    @staticmethod
    def create_progress_bar(current: int, total: int, label: str):
        """Создание прогресс-бара"""
        progress = current / total if total > 0 else 0
        st.progress(progress, text=f"{label}: {current}/{total} ({progress*100:.1f}%)")

class ChartComponents:
    """Компоненты для создания графиков"""

    @staticmethod
    def create_phase_comparison_chart(data: Dict, title: str = "Сравнение фаз"):
        """График сравнения значений по фазам"""
        phases = ['R', 'S', 'T']
        metrics = ['RMS', 'Crest Factor', 'Kurtosis', 'Skewness']

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=metrics,
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        colors = ['red', 'green', 'blue']

        for i, metric in enumerate(metrics):
            row = i // 2 + 1
            col = i % 2 + 1

            metric_key = metric.lower().replace(' ', '_')
            values = []

            for phase_idx, phase in enumerate(['a', 'b', 'c']):
                key = f"{metric_key}_{phase}"
                if key in data:
                    values.append(data[key])
                else:
                    values.append(0)

            fig.add_trace(
                go.Bar(
                    x=phases,
                    y=values,
                    name=metric,
                    marker_color=colors,
                    showlegend=False
                ),
                row=row, col=col
            )

        fig.update_layout(
            title=title,
            height=600,
            showlegend=False
        )

        return fig

    @staticmethod
    def create_anomaly_timeline(anomalies: List[Dict], title: str = "Временная линия аномалий"):
        """График временной линии аномалий"""
        if not anomalies:
            return None

        # Подготовка данных
        timeline_data = []
        for anomaly in anomalies:
            timeline_data.append({
                'timestamp': anomaly.get('created_at', ''),
                'type': anomaly.get('defect_type', 'Unknown'),
                'probability': anomaly.get('probability', 0),
                'severity': anomaly.get('predicted_severity', 'Low'),
                'is_anomaly': anomaly.get('is_anomaly', False)
            })

        df = pd.DataFrame(timeline_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Создание графика
        fig = go.Figure()

        # Нормальные точки
        normal_points = df[~df['is_anomaly']]
        if not normal_points.empty:
            fig.add_trace(
                go.Scatter(
                    x=normal_points['timestamp'],
                    y=normal_points['probability'],
                    mode='markers',
                    marker=dict(color='green', size=8),
                    name='Нормальное состояние',
                    hovertemplate='Время: %{x}<br>Вероятность: %{y:.3f}<extra></extra>'
                )
            )

        # Аномальные точки
        anomaly_points = df[df['is_anomaly']]
        if not anomaly_points.empty:
            fig.add_trace(
                go.Scatter(
                    x=anomaly_points['timestamp'],
                    y=anomaly_points['probability'],
                    mode='markers',
                    marker=dict(
                        color='red',
                        size=12,
                        symbol='triangle-up'
                    ),
                    name='Аномалия',
                    hovertemplate='Время: %{x}<br>Вероятность: %{y:.3f}<br>Тип: %{customdata}<extra></extra>',
                    customdata=anomaly_points['type']
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Время",
            yaxis_title="Вероятность аномалии",
            height=400,
            hovermode='x unified'
        )

        return fig

    @staticmethod
    def create_health_gauge(health_score: float, title: str = "Индекс здоровья оборудования"):
        """Создание круговой диаграммы состояния"""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = health_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': title},
            delta = {'reference': 100},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))

        fig.update_layout(height=300)
        return fig

class FilterComponents:
    """Компоненты для фильтрации данных"""

    @staticmethod
    def create_date_filter(key: str = "date_filter"):
        """Создание фильтра по датам"""
        col1, col2 = st.columns(2)

        with col1:
            start_date = st.date_input(
                "Дата начала",
                value=datetime.now() - timedelta(days=30),
                key=f"{key}_start"
            )

        with col2:
            end_date = st.date_input(
                "Дата окончания",
                value=datetime.now(),
                key=f"{key}_end"
            )

        return start_date, end_date

    @staticmethod
    def create_equipment_filter(equipment_list: List[Dict], key: str = "equipment_filter"):
        """Создание фильтра по оборудованию"""
        equipment_options = ["Все"] + [eq['name'] for eq in equipment_list]
        selected = st.selectbox(
            "Фильтр по оборудованию:",
            options=equipment_options,
            key=key
        )

        if selected == "Все":
            return None

        return next((eq for eq in equipment_list if eq['name'] == selected), None)

    @staticmethod
    def create_severity_filter(key: str = "severity_filter"):
        """Создание фильтра по серьезности"""
        severity_options = ["Все", "Low", "Medium", "High", "Critical"]
        return st.selectbox(
            "Фильтр по серьезности:",
            options=severity_options,
            key=key
        )

class ExportComponents:
    """Компоненты для экспорта данных"""

    @staticmethod
    def create_csv_download(data: pd.DataFrame, filename: str, label: str = "Скачать CSV"):
        """Создание кнопки скачивания CSV"""
        csv = data.to_csv(index=False)
        st.download_button(
            label=label,
            data=csv,
            file_name=f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    @staticmethod
    def create_json_download(data: Dict, filename: str, label: str = "Скачать JSON"):
        """Создание кнопки скачивания JSON"""
        import json
        json_str = json.dumps(data, ensure_ascii=False, indent=2)
        st.download_button(
            label=label,
            data=json_str,
            file_name=f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

class NotificationComponents:
    """Компоненты для уведомлений"""

    @staticmethod
    def show_success(message: str, icon: str = "✅"):
        """Показать сообщение об успехе"""
        st.success(f"{icon} {message}")

    @staticmethod
    def show_warning(message: str, icon: str = "⚠️"):
        """Показать предупреждение"""
        st.warning(f"{icon} {message}")

    @staticmethod
    def show_error(message: str, icon: str = "❌"):
        """Показать ошибку"""
        st.error(f"{icon} {message}")

    @staticmethod
    def show_info(message: str, icon: str = "ℹ️"):
        """Показать информацию"""
        st.info(f"{icon} {message}")

    @staticmethod
    def show_alert_banner(severity: str, title: str, message: str):
        """Показать баннер с предупреждением"""
        alert_styles = {
            "critical": {"color": "#721c24", "bg": "#f8d7da"},
            "warning": {"color": "#856404", "bg": "#fff3cd"},
            "info": {"color": "#0c5460", "bg": "#d1ecf1"}
        }

        style = alert_styles.get(severity, alert_styles["info"])

        st.markdown(
            f"""
            <div style="
                padding: 1rem;
                margin: 1rem 0;
                border-radius: 0.375rem;
                background-color: {style['bg']};
                color: {style['color']};
                border: 1px solid {style['color']};
            ">
                <strong>{title}</strong><br>
                {message}
            </div>
            """,
            unsafe_allow_html=True
        )

class ConfigComponents:
    """Компоненты для настройки"""

    @staticmethod
    def create_settings_panel():
        """Создание панели настроек"""
        with st.expander("⚙️ Настройки отображения"):
            col1, col2 = st.columns(2)

            with col1:
                chart_theme = st.selectbox(
                    "Тема графиков:",
                    ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"]
                )

                auto_refresh = st.checkbox("Автообновление", value=False)

            with col2:
                refresh_interval = st.slider(
                    "Интервал обновления (сек):",
                    min_value=10,
                    max_value=300,
                    value=60,
                    disabled=not auto_refresh
                )

                show_tooltips = st.checkbox("Показывать подсказки", value=True)

            return {
                "chart_theme": chart_theme,
                "auto_refresh": auto_refresh,
                "refresh_interval": refresh_interval,
                "show_tooltips": show_tooltips
            }
