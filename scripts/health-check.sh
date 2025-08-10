#!/bin/bash
# =============================================================================
# DiagMod Health Check Script
# Проверка работоспособности всех сервисов
# =============================================================================

set -euo pipefail

# Цвета для вывода
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Функции
log() { echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"; }
success() { echo -e "${GREEN}✓${NC} $1"; }
error() { echo -e "${RED}✗${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC} $1"; }

# Проверка Docker
check_docker() {
    log "Проверка Docker..."
    if docker info &>/dev/null; then
        success "Docker работает"
    else
        error "Docker недоступен"
        return 1
    fi
}

# Проверка сервисов
check_services() {
    log "Проверка статуса контейнеров..."

    local services=("postgres" "redis" "api" "worker" "dashboard" "prometheus" "grafana")
    local failed=0

    for service in "${services[@]}"; do
        if docker-compose ps "$service" | grep -q "Up"; then
            success "$service запущен"
        else
            error "$service не запущен"
            ((failed++))
        fi
    done

    return $failed
}

# Проверка API endpoints
check_endpoints() {
    log "Проверка API endpoints..."

    # API Health
    if curl -f -s http://localhost:8000/health >/dev/null 2>&1; then
        success "API здоров (http://localhost:8000/health)"
    else
        error "API недоступен"
    fi

    # Dashboard
    if curl -f -s http://localhost:8501 >/dev/null 2>&1; then
        success "Dashboard доступен (http://localhost:8501)"
    else
        error "Dashboard недоступен"
    fi

    # Prometheus
    if curl -f -s http://localhost:9090/-/healthy >/dev/null 2>&1; then
        success "Prometheus здоров (http://localhost:9090)"
    else
        error "Prometheus недоступен"
    fi

    # Grafana
    if curl -f -s http://localhost:3000/api/health >/dev/null 2>&1; then
        success "Grafana здорова (http://localhost:3000)"
    else
        error "Grafana недоступна"
    fi
}

# Проверка базы данных
check_database() {
    log "Проверка базы данных..."

    if docker-compose exec -T postgres pg_isready -U diagmod_user -d diagmod >/dev/null 2>&1; then
        success "PostgreSQL готова к работе"
    else
        error "PostgreSQL недоступна"
    fi
}

# Проверка Redis
check_redis() {
    log "Проверка Redis..."

    if docker-compose exec -T redis redis-cli ping >/dev/null 2>&1; then
        success "Redis отвечает"
    else
        error "Redis недоступен"
    fi
}

# Проверка Celery
check_celery() {
    log "Проверка Celery worker..."

    if docker-compose exec -T worker celery -A src.worker.celery_app inspect ping >/dev/null 2>&1; then
        success "Celery worker активен"
    else
        error "Celery worker недоступен"
    fi
}

# Главная функция
main() {
    echo -e "${BLUE}=== DiagMod Health Check ===${NC}"
    echo ""

    local exit_code=0

    check_docker || exit_code=1
    check_services || exit_code=1
    check_database || exit_code=1
    check_redis || exit_code=1
    check_celery || exit_code=1
    check_endpoints || exit_code=1

    echo ""
    if [ $exit_code -eq 0 ]; then
        success "Все сервисы работают корректно!"
    else
        error "Обнаружены проблемы в работе сервисов"
        echo "Запустите 'make logs' для диагностики"
    fi

    exit $exit_code
}

main "$@"
