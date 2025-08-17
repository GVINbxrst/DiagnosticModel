"""Страница трендов: почасовой RMS и риск (sequence LSTM)"""
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from datetime import datetime, timedelta

def render_trends(api_base_url: str, token: str, equipment_id: str):
    st.subheader("📈 Почасовые тренды RMS")
    headers = {"Authorization": f"Bearer {token}"}
    limit = st.slider("Горизонт (часов)", min_value=24, max_value=24*14, value=24*7, step=24)
    with st.spinner("Загрузка агрегатов..."):
        try:
            r = requests.get(f"{api_base_url}/api/v1/equipment/{equipment_id}/rms/hourly", params={"limit": limit}, headers=headers, timeout=30)
            data = r.json().get('points', []) if r.status_code == 200 else []
        except Exception as e:
            st.error(f"Ошибка загрузки агрегатов: {e}")
            data = []
    if not data:
        st.info("Нет агрегированных данных")
        return
    df = pd.DataFrame(data)
    # Совместимость ключей
    if 'timestamp' in df.columns:
        df['ts'] = pd.to_datetime(df['timestamp'])
    elif 'ts' in df.columns:
        df['ts'] = pd.to_datetime(df['ts'])
    df = df.sort_values('ts')
    fig = px.line(df, x='ts', y='rms_mean', title='RMS (mean) по часам')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🛡️ Прогноз риска (sequence model)")
    horizon = st.slider("Горизонт прогноза (окон)", min_value=3, max_value=24, value=5)
    if st.button("Рассчитать риск", type="primary"):
        with st.spinner("Прогнозирование риска..."):
            try:
                r = requests.get(f"{api_base_url}/api/v1/signals/sequence_risk/{equipment_id}", params={"horizon": horizon}, headers=headers, timeout=60)
                if r.status_code == 200:
                    res = r.json()
                    st.metric("Risk score", f"{res.get('risk_score',0):.3f}")
                else:
                    st.error(f"Ошибка прогноза: {r.text}")
            except Exception as e:
                st.error(f"Ошибка прогноза: {e}")

    st.subheader("🔮 Прогноз RMS (Prophet/Fallback)")
    fc_steps = st.slider("Шагов вперёд (часов)", min_value=6, max_value=72, value=24, step=6)
    if st.button("Построить прогноз RMS"):
        with st.spinner("Запрос прогноза..."):
            try:
                # Предполагается существование эндпоинта forecast_rms (можно реализовать аналогично)
                # Основной (канонический) путь /api/v1/forecast_rms/{equipment_id}; оставлен серверный алиас на /anomalies/forecast_rms/
                r = requests.get(f"{api_base_url}/api/v1/forecast_rms/{equipment_id}", params={"steps": fc_steps}, headers=headers, timeout=120)
                if r.status_code == 404:
                    # Fallback на старый путь, если развернута старая версия API
                    r = requests.get(f"{api_base_url}/api/v1/anomalies/forecast_rms/{equipment_id}", params={"steps": fc_steps}, headers=headers, timeout=120)
                if r.status_code == 200:
                    data = r.json()
                    fc = data.get('forecast', [])
                    import pandas as pd
                    if fc:
                        df_fc = pd.DataFrame(fc)
                        import plotly.express as px
                        fig_fc = px.line(df_fc, x='timestamp', y='rms', title='Прогноз RMS')
                        # Порог
                        thr = data.get('threshold')
                        if thr is not None:
                            import plotly.graph_objects as go
                            fig_fc.add_hline(y=thr, line_dash='dash', line_color='red', annotation_text='threshold')
                        st.plotly_chart(fig_fc, use_container_width=True)
                        st.caption(f"Модель: {data.get('model')} ProbOver: {data.get('probability_over_threshold'):.3f}")
                    else:
                        st.info("Нет данных прогноза")
                else:
                    st.error(f"Ошибка прогноза: {r.text}")
            except Exception as e:
                st.error(f"Ошибка запроса прогноза: {e}")
