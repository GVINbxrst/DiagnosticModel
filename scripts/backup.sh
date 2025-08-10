#!/bin/bash
# =============================================================================
# DiagMod Backup Script
# Создание резервных копий базы данных и важных данных
# =============================================================================

set -euo pipefail

# Конфигурация
BACKUP_DIR="/backups"
DB_NAME="${PGDATABASE:-diagmod}"
DB_USER="${PGUSER:-diagmod_user}"
DB_HOST="${PGHOST:-postgres}"
RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-30}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Функция логирования
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" >&2
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

# Проверка доступности PostgreSQL
check_postgres() {
    log "Проверка доступности PostgreSQL..."
    if ! pg_isready -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -q; then
        error "PostgreSQL недоступен!"
        exit 1
    fi
    log "PostgreSQL доступен"
}

# Создание директории для бэкапов
create_backup_dir() {
    log "Создание директории для бэкапов..."
    mkdir -p "$BACKUP_DIR/database"
    mkdir -p "$BACKUP_DIR/data"
    mkdir -p "$BACKUP_DIR/models"
    mkdir -p "$BACKUP_DIR/configs"
}

# Резервное копирование базы данных
backup_database() {
    log "Создание резервной копии базы данных..."

    local backup_file="$BACKUP_DIR/database/diagmod_${TIMESTAMP}.sql"
    local backup_file_gz="${backup_file}.gz"

    # Создание дампа
    if pg_dump -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" \
        --clean --create --if-exists --verbose \
        --format=plain > "$backup_file"; then

        # Сжатие дампа
        gzip "$backup_file"
        log "Резервная копия БД создана: $backup_file_gz"

        # Создание метаданных
        cat > "${backup_file_gz}.meta" << EOF
{
    "timestamp": "$TIMESTAMP",
    "database": "$DB_NAME",
    "host": "$DB_HOST",
    "user": "$DB_USER",
    "size_bytes": $(stat -c%s "$backup_file_gz"),
    "created_at": "$(date -Iseconds)",
    "retention_until": "$(date -d "+$RETENTION_DAYS days" -Iseconds)"
}
EOF

    else
        error "Ошибка создания резервной копии БД"
        exit 1
    fi
}

# Резервное копирование данных
backup_data() {
    log "Создание резервной копии данных..."

    local data_backup="$BACKUP_DIR/data/data_${TIMESTAMP}.tar.gz"

    # Архивирование папки data (исключая временные файлы)
    if tar -czf "$data_backup" \
        --exclude='*.tmp' \
        --exclude='*.log' \
        --exclude='__pycache__' \
        -C /app data/ 2>/dev/null; then

        log "Резервная копия данных создана: $data_backup"
    else
        warn "Ошибка создания резервной копии данных (возможно, папка пуста)"
    fi
}

# Резервное копирование моделей ML
backup_models() {
    log "Создание резервной копии ML моделей..."

    local models_backup="$BACKUP_DIR/models/models_${TIMESTAMP}.tar.gz"

    # Архивирование папки models
    if tar -czf "$models_backup" \
        --exclude='*.tmp' \
        --exclude='__pycache__' \
        -C /app models/ 2>/dev/null; then

        log "Резервная копия моделей создана: $models_backup"
    else
        warn "Ошибка создания резервной копии моделей (возможно, папка пуста)"
    fi
}

# Резервное копирование конфигураций
backup_configs() {
    log "Создание резервной копии конфигураций..."

    local config_backup="$BACKUP_DIR/configs/configs_${TIMESTAMP}.tar.gz"

    # Копирование важных конфигурационных файлов
    if tar -czf "$config_backup" \
        -C /app \
        pyproject.toml \
        requirements.txt \
        docker-compose.yml \
        .env.example 2>/dev/null; then

        log "Резервная копия конфигураций создана: $config_backup"
    else
        warn "Ошибка создания резервной копии конфигураций"
    fi
}

# Очистка старых резервных копий
cleanup_old_backups() {
    log "Очистка старых резервных копий (старше $RETENTION_DAYS дней)..."

    local deleted_count=0

    # Поиск и удаление старых файлов
    for backup_type in database data models configs; do
        while IFS= read -r -d '' file; do
            rm -f "$file"
            ((deleted_count++))
            log "Удален старый бэкап: $(basename "$file")"
        done < <(find "$BACKUP_DIR/$backup_type" -name "*" -type f -mtime +$RETENTION_DAYS -print0 2>/dev/null)
    done

    if [ $deleted_count -eq 0 ]; then
        log "Старых резервных копий для удаления не найдено"
    else
        log "Удалено старых резервных копий: $deleted_count"
    fi
}

# Проверка места на диске
check_disk_space() {
    log "Проверка места на диске..."

    local available_space=$(df "$BACKUP_DIR" | awk 'NR==2 {print $4}')
    local available_gb=$((available_space / 1024 / 1024))

    if [ $available_gb -lt 1 ]; then
        error "Недостаточно места на диске: ${available_gb}GB доступно"
        exit 1
    fi

    log "Доступно места на диске: ${available_gb}GB"
}

# Создание общего архива
create_full_backup() {
    log "Создание полного архива..."

    local full_backup="$BACKUP_DIR/full_backup_${TIMESTAMP}.tar.gz"

    if tar -czf "$full_backup" \
        -C "$BACKUP_DIR" \
        database/diagmod_${TIMESTAMP}.sql.gz \
        data/data_${TIMESTAMP}.tar.gz \
        models/models_${TIMESTAMP}.tar.gz \
        configs/configs_${TIMESTAMP}.tar.gz 2>/dev/null; then

        log "Полный архив создан: $full_backup"

        # Создание манифеста
        cat > "${full_backup}.manifest" << EOF
DiagMod Full Backup Manifest
=============================
Timestamp: $TIMESTAMP
Created: $(date -Iseconds)
Components:
- Database dump (compressed)
- Application data
- ML models
- Configuration files

To restore:
1. Extract: tar -xzf $(basename "$full_backup")
2. Restore DB: gunzip -c database/diagmod_${TIMESTAMP}.sql.gz | psql
3. Restore data: tar -xzf data/data_${TIMESTAMP}.tar.gz
4. Restore models: tar -xzf models/models_${TIMESTAMP}.tar.gz
EOF

    else
        error "Ошибка создания полного архива"
        exit 1
    fi
}

# Основная функция
main() {
    log "Начало создания резервной копии DiagMod"
    log "Timestamp: $TIMESTAMP"

    check_disk_space
    check_postgres
    create_backup_dir

    backup_database
    backup_data
    backup_models
    backup_configs

    create_full_backup
    cleanup_old_backups

    log "Резервное копирование завершено успешно!"
    log "Архивы сохранены в: $BACKUP_DIR"
}

# Обработка сигналов
trap 'error "Резервное копирование прервано"; exit 1' INT TERM

# Запуск
main "$@"
