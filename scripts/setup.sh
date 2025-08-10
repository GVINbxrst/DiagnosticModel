#!/bin/bash

# DiagMod Setup Script
# Скрипт автоматической установки и настройки системы диагностики двигателей

set -e  # Остановка при любой ошибке

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Функции для вывода
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Проверка системы
check_system() {
    print_info "Проверка системных требований..."

    # Проверяем Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 не найден. Установите Python 3.8 или выше."
        exit 1
    fi

    python_version=$(python3 --version | cut -d' ' -f2)
    print_info "Найден Python $python_version"

    # Проверяем pip
    if ! command -v pip3 &> /dev/null; then
        print_error "pip3 не найден. Установите pip."
        exit 1
    fi

    # Проверяем Docker
    if ! command -v docker &> /dev/null; then
        print_warning "Docker не найден. Для полноценной работы установите Docker."
    fi

    # Проверяем Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_warning "Docker Compose не найден. Для полноценной работы установите Docker Compose."
    fi

    print_success "Системные требования проверены"
}

# Создание виртуального окружения
setup_venv() {
    print_info "Создание виртуального окружения..."

    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Виртуальное окружение создано"
    else
        print_info "Виртуальное окружение уже существует"
    fi

    # Активация виртуального окружения
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        # Windows
        source venv/Scripts/activate
    else
        # Unix/Linux/Mac
        source venv/bin/activate
    fi

    print_success "Виртуальное окружение активировано"
}

# Установка зависимостей
install_dependencies() {
    print_info "Установка зависимостей..."

    # Обновляем pip, setuptools, wheel
    pip3 install --upgrade pip setuptools wheel
    print_success "Базовые инструменты обновлены"

    # Устанавливаем основные зависимости
    pip3 install -r requirements.txt
    print_success "Основные зависимости установлены"

    # Устанавливаем зависимости для разработки (опционально)
    if [ "$INSTALL_DEV" = "true" ]; then
        pip3 install -r requirements-dev.txt
        print_success "Зависимости для разработки установлены"
    fi
}

# Создание директорий
create_directories() {
    print_info "Создание необходимых директорий..."

    mkdir -p logs
    mkdir -p data/{raw,processed,features,exports}
    mkdir -p models/{anomaly_detection,prediction}
    mkdir -p configs/ssl

    print_success "Директории созданы"
}

# Настройка конфигурации
setup_config() {
    print_info "Настройка конфигурации..."

    # Создаем .env файл если его нет
    if [ ! -f ".env" ]; then
        cat > .env << EOF
# DiagMod Environment Configuration

# Database
DATABASE_URL=postgresql+asyncpg://diagmod_user:diagmod_password@localhost:5432/diagmod

# Redis
REDIS_URL=redis://localhost:6379/0

# Celery
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2

# API
API_HOST=0.0.0.0
API_PORT=8000

# Security
SECRET_KEY=your-secret-key-change-in-production
JWT_SECRET_KEY=your-jwt-secret-key-change-in-production

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE_PATH=logs/diagmod.log

# Monitoring
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=8001
WORKER_METRICS_PORT=8002

# Environment
ENVIRONMENT=development
EOF
        print_success "Файл .env создан"
    else
        print_info "Файл .env уже существует"
    fi
}

# Инициализация базы данных (если Docker запущен)
init_database() {
    if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
        print_info "Попытка инициализации базы данных..."

        # Проверяем, запущен ли PostgreSQL в Docker
        if docker-compose ps postgres | grep -q "Up"; then
            python3 -c "
import asyncio
from src.database.connection import init_database
asyncio.run(init_database())
print('База данных инициализирована')
" && print_success "База данных инициализирована" || print_warning "Не удалось инициализировать базу данных"
        else
            print_warning "PostgreSQL не запущен. Запустите 'make docker-up' для инициализации БД"
        fi
    else
        print_warning "Docker не найден. База данных не инициализирована"
    fi
}

# Проверка установки
verify_installation() {
    print_info "Проверка установки..."

    # Проверяем импорт основных модулей
    python3 -c "
try:
    import fastapi
    import streamlit
    import celery
    import sqlalchemy
    import numpy
    import pandas
    import sklearn
    import prometheus_client
    print('✓ Все основные модули доступны')
except ImportError as e:
    print(f'✗ Ошибка импорта: {e}')
    exit(1)
"

    print_success "Установка проверена"
}

# Вывод инструкций
show_instructions() {
    print_success "Установка завершена!"
    echo ""
    print_info "Следующие шаги:"
    echo ""
    echo "1. Активируйте виртуальное окружение:"
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        echo "   source venv/Scripts/activate"
    else
        echo "   source venv/bin/activate"
    fi
    echo ""
    echo "2. Запустите Docker сервисы:"
    echo "   make docker-up"
    echo ""
    echo "3. Инициализируйте базу данных:"
    echo "   make db-init"
    echo ""
    echo "4. Запустите компоненты для разработки:"
    echo "   make dev-api      # API сервер"
    echo "   make dev-worker   # Worker"
    echo "   make dev-dashboard # Dashboard"
    echo ""
    echo "Или запустите всё сразу в Docker:"
    echo "   make docker-up"
    echo ""
    print_info "Доступные команды:"
    echo "   make help         # Показать все команды"
    echo "   make info         # Информация о прое��те"
    echo "   make status       # Статус сервисов"
    echo ""
    print_info "Веб-интерфейсы (после запуска):"
    echo "   http://localhost:8000     # API"
    echo "   http://localhost:8501     # Dashboard"
    echo "   http://localhost:9090     # Prometheus"
    echo ""
}

# Основная функция
main() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                    DiagMod Setup - Установка системы                         ║"
    echo "║                  Диагностика асинхронных двигателей                          ║"
    echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"

    # Парсинг аргументов
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dev)
                INSTALL_DEV="true"
                shift
                ;;
            --help)
                echo "Использование: $0 [опции]"
                echo ""
                echo "Опции:"
                echo "  --dev     Установить зависимости для разработки"
                echo "  --help    Показать эту справку"
                exit 0
                ;;
            *)
                print_error "Неизвестная опция: $1"
                exit 1
                ;;
        esac
    done

    # Выполняем установку
    check_system
    setup_venv
    install_dependencies
    create_directories
    setup_config
    verify_installation

    # Пытаемся инициализировать БД если возможно
    init_database

    show_instructions
}

# Запуск основной функции
main "$@"
