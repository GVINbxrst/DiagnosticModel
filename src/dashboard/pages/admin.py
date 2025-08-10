"""
Страница администрирования для Dashboard
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import requests
import logging

logger = logging.getLogger(__name__)

class AdminPage:
    """Страница администрирования системы"""

    def __init__(self, data_manager, user_info: Dict):
        self.data_manager = data_manager
        self.user_info = user_info
        self.is_admin = user_info.get('role') == 'admin'

    def render(self):
        """Отображение страницы администрирования"""
        if not self.is_admin:
            st.error("🔒 Доступ запрещен. Требуются права администратора.")
            return

        st.title("🛠️ Администрирование системы")

        # Вкладки администрирования
        tab1, tab2, tab3, tab4 = st.tabs([
            "👥 Пользователи",
            "🖥️ Система",
            "📊 Статистика",
            "🔐 Безопасность"
        ])

        with tab1:
            self._render_users_management()

        with tab2:
            self._render_system_status()

        with tab3:
            self._render_statistics()

        with tab4:
            self._render_security_audit()

    def _render_users_management(self):
        """Управление пользователями"""
        st.subheader("👥 Управление пользователями")

        # Получение списка пользователей
        users = self._get_users_list()

        if users:
            # Статистика пользователей
            col1, col2, col3, col4 = st.columns(4)

            total_users = len(users)
            active_users = len([u for u in users if u.get('is_active', False)])
            admin_users = len([u for u in users if u.get('role') == 'admin'])
            recent_logins = len([u for u in users if self._is_recent_login(u.get('last_login'))])

            with col1:
                st.metric("Всего пользователей", total_users)
            with col2:
                st.metric("Активных", active_users)
            with col3:
                st.metric("Администраторов", admin_users)
            with col4:
                st.metric("Недавние входы", recent_logins)

            # Таблица пользователей
            df_users = pd.DataFrame(users)
            st.dataframe(df_users, use_container_width=True)

            # Действия с пользователями
            st.subheader("Действия")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("➕ Добавить пользователя"):
                    self._show_add_user_form()

            with col2:
                selected_user = st.selectbox(
                    "Выберите пользователя для действий:",
                    options=[f"{u['username']} ({u['role']})" for u in users]
                )

                if selected_user and st.button("🔧 Управление"):
                    username = selected_user.split(' ')[0]
                    self._show_user_management(username)
        else:
            st.info("Пользователи не найдены")

    def _render_system_status(self):
        """Состояние системы"""
        st.subheader("🖥️ Состояние системы")

        # Проверка состояния сервисов
        services_status = self._check_services_status()

        # Отображение статуса сервисов
        for service, status in services_status.items():
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.write(f"**{service}**")

            with col2:
                if status['healthy']:
                    st.success("🟢 Работает")
                else:
                    st.error("🔴 Недоступен")

            with col3:
                if 'response_time' in status:
                    st.write(f"{status['response_time']:.2f}ms")

        # Системные метрики
        st.subheader("📈 Системные метрики")

        metrics = self._get_system_metrics()
        if metrics:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("CPU", f"{metrics.get('cpu_usage', 0):.1f}%")
            with col2:
                st.metric("RAM", f"{metrics.get('memory_usage', 0):.1f}%")
            with col3:
                st.metric("Диск", f"{metrics.get('disk_usage', 0):.1f}%")
            with col4:
                st.metric("Активные соединения", metrics.get('active_connections', 0))

    def _render_statistics(self):
        """Статистика использования"""
        st.subheader("📊 Статистика использования")

        # Период анализа
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Начало периода", value=datetime.now() - timedelta(days=7))
        with col2:
            end_date = st.date_input("Конец периода", value=datetime.now())

        # Получение статистики
        stats = self._get_usage_statistics(start_date, end_date)

        if stats:
            # Основные метрики
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Загружено файлов", stats.get('files_uploaded', 0))
            with col2:
                st.metric("Обработано сигналов", stats.get('signals_processed', 0))
            with col3:
                st.metric("Найдено аномалий", stats.get('anomalies_detected', 0))
            with col4:
                st.metric("API запросов", stats.get('api_requests', 0))

            # График активности по дням
            if 'daily_activity' in stats:
                fig = px.line(
                    x=[d['date'] for d in stats['daily_activity']],
                    y=[d['requests'] for d in stats['daily_activity']],
                    title="Активность по дням"
                )
                st.plotly_chart(fig, use_container_width=True)

            # График распределения типов аномалий
            if 'anomaly_types' in stats:
                fig = px.pie(
                    values=list(stats['anomaly_types'].values()),
                    names=list(stats['anomaly_types'].keys()),
                    title="Распределение типов аномалий"
                )
                st.plotly_chart(fig, use_container_width=True)

    def _render_security_audit(self):
        """Аудит безопасности"""
        st.subheader("🔐 Аудит безопасности")

        # Недавние события безопасности
        security_events = self._get_security_events()

        if security_events:
            st.subheader("Недавние события")

            # Статистика событий
            col1, col2, col3, col4 = st.columns(4)

            failed_logins = len([e for e in security_events if e.get('event_type') == 'failed_login'])
            successful_logins = len([e for e in security_events if e.get('event_type') == 'successful_login'])
            permission_denied = len([e for e in security_events if e.get('event_type') == 'permission_denied'])
            suspicious_activity = len([e for e in security_events if e.get('severity') == 'high'])

            with col1:
                st.metric("Неудачные входы", failed_logins, delta=f"За последние 24ч")
            with col2:
                st.metric("Успешные входы", successful_logins)
            with col3:
                st.metric("Отказы доступа", permission_denied)
            with col4:
                st.metric("Подозрительная активность", suspicious_activity, delta_color="inverse")

            # Таблица событий
            df_events = pd.DataFrame(security_events)
            if not df_events.empty:
                st.dataframe(df_events, use_container_width=True)

        # Настройки безопасности
        st.subheader("Настройки безопасности")

        col1, col2 = st.columns(2)

        with col1:
            max_login_attempts = st.number_input("Макс. попыток входа", min_value=1, max_value=10, value=5)
            session_timeout = st.number_input("Таймаут сессии (мин)", min_value=15, max_value=480, value=60)

        with col2:
            require_2fa = st.checkbox("Требовать 2FA", value=False)
            log_all_requests = st.checkbox("Логировать все запросы", value=True)

        if st.button("💾 Сохранить настройки"):
            self._save_security_settings({
                'max_login_attempts': max_login_attempts,
                'session_timeout': session_timeout,
                'require_2fa': require_2fa,
                'log_all_requests': log_all_requests
            })
            st.success("Настройки сохранены")

    def _get_users_list(self) -> List[Dict]:
        """Получение списка пользователей"""
        try:
            response = requests.get(
                f"{self.data_manager.API_BASE_URL}/admin/users",
                headers=self.data_manager.headers
            )
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            logger.error(f"Ошибка получения пользователей: {e}")
            return []

    def _is_recent_login(self, last_login: Optional[str]) -> bool:
        """Проверка недавнего входа"""
        if not last_login:
            return False

        try:
            login_date = datetime.fromisoformat(last_login.replace('Z', '+00:00'))
            return (datetime.now() - login_date).days <= 7
        except:
            return False

    def _check_services_status(self) -> Dict:
        """Проверка состояния сервисов"""
        services = {
            'API': f"{self.data_manager.API_BASE_URL}/health",
            'База данных': f"{self.data_manager.API_BASE_URL}/health/db",
            'Redis': f"{self.data_manager.API_BASE_URL}/health/redis",
            'Worker': f"{self.data_manager.API_BASE_URL}/health/worker"
        }

        status = {}
        for service, url in services.items():
            try:
                start_time = datetime.now()
                response = requests.get(url, timeout=5)
                end_time = datetime.now()

                status[service] = {
                    'healthy': response.status_code == 200,
                    'response_time': (end_time - start_time).total_seconds() * 1000
                }
            except Exception:
                status[service] = {
                    'healthy': False,
                    'response_time': None
                }

        return status

    def _get_system_metrics(self) -> Optional[Dict]:
        """Получение системных метрик"""
        try:
            response = requests.get(
                f"{self.data_manager.API_BASE_URL}/admin/metrics",
                headers=self.data_manager.headers
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"Ошибка получения метрик: {e}")
            return None

    def _get_usage_statistics(self, start_date, end_date) -> Optional[Dict]:
        """Получение статистики использования"""
        try:
            response = requests.get(
                f"{self.data_manager.API_BASE_URL}/admin/statistics",
                headers=self.data_manager.headers,
                params={
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat()
                }
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            return None

    def _get_security_events(self) -> List[Dict]:
        """Получение событий безопасности"""
        try:
            response = requests.get(
                f"{self.data_manager.API_BASE_URL}/admin/security/events",
                headers=self.data_manager.headers
            )
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            logger.error(f"Ошибка получения событий безопасности: {e}")
            return []

    def _save_security_settings(self, settings: Dict):
        """Сохранение настроек безопасности"""
        try:
            response = requests.post(
                f"{self.data_manager.API_BASE_URL}/admin/security/settings",
                headers=self.data_manager.headers,
                json=settings
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ошибка сохранения настроек: {e}")
            return False

    def _show_add_user_form(self):
        """Форма добавления пользователя"""
        with st.form("add_user_form"):
            st.subheader("Добавить нового пользователя")

            col1, col2 = st.columns(2)

            with col1:
                username = st.text_input("Имя пользователя")
                email = st.text_input("Email")
                password = st.text_input("Пароль", type="password")

            with col2:
                role = st.selectbox("Роль", ["operator", "engineer", "admin"])
                is_active = st.checkbox("Активен", value=True)
                department = st.text_input("Отдел")

            submitted = st.form_submit_button("Создать пользователя")

            if submitted:
                if username and email and password:
                    user_data = {
                        'username': username,
                        'email': email,
                        'password': password,
                        'role': role,
                        'is_active': is_active,
                        'department': department
                    }

                    if self._create_user(user_data):
                        st.success("Пользователь создан успешно")
                        st.rerun()
                    else:
                        st.error("Ошибка создания пользователя")
                else:
                    st.error("Заполните обязательные поля")

    def _create_user(self, user_data: Dict) -> bool:
        """Создание пользователя"""
        try:
            response = requests.post(
                f"{self.data_manager.API_BASE_URL}/admin/users",
                headers=self.data_manager.headers,
                json=user_data
            )
            return response.status_code == 201
        except Exception as e:
            logger.error(f"Ошибка создания пользователя: {e}")
            return False
