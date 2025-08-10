#!/bin/bash

# Скрипт запуска Dashboard в режиме разработки

set -e

echo "🚀 Запуск DiagMod Dashboard..."

# Проверка переменных окружения
if [ -z "$API_BASE_URL" ]; then
    export API_BASE_URL="http://localhost:8000"
    echo "⚠️  Используется API_BASE_URL по умолчанию: $API_BASE_URL"
fi

# Установка переменных Streamlit
export STREAMLIT_SERVER_PORT=${DASHBOARD_PORT:-8501}
export STREAMLIT_SERVER_ADDRESS=${DASHBOARD_HOST:-0.0.0.0}
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Создание необходимых директорий
mkdir -p logs/.streamlit

# Копирование конфигурации Streamlit
if [ -f "configs/streamlit/config.toml" ]; then
    cp configs/streamlit/config.toml ~/.streamlit/config.toml
    echo "✅ Конфигурация Streamlit скопирована"
fi

# Запуск приложения
echo "🌐 Dashboard будет доступен по адресу: http://$STREAMLIT_SERVER_ADDRESS:$STREAMLIT_SERVER_PORT"
echo "🔗 API подключение: $API_BASE_URL"

exec streamlit run src/dashboard/main.py \
    --server.port=$STREAMLIT_SERVER_PORT \
    --server.address=$STREAMLIT_SERVER_ADDRESS \
    --server.headless=true \
    --browser.gatherUsageStats=false
