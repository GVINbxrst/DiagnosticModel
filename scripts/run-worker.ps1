# DiagMod Worker - PowerShell Launch Script
# Запуск Celery Worker для фоновых задач

param(
    [string]$LogLevel = "info",
    [int]$Concurrency = 4,
    [string]$Queue = "",
    [switch]$Beat = $false,
    [switch]$Flower = $false,
    [switch]$Help
)

if ($Help) {
    Write-Host "DiagMod Celery Worker" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Параметры:"
    Write-Host "  -LogLevel <level>    Уровень логирования (debug, info, warning, error)"
    Write-Host "  -Concurrency <num>   Количество воркер процессов (по умолчанию: 4)"
    Write-Host "  -Queue <name>        Имя очереди для обработки"
    Write-Host "  -Beat               Запустить Celery Beat (планировщик задач)"
    Write-Host "  -Flower             Запустить Flower (веб-интерфейс мониторинга)"
    Write-Host "  -Help               Показать эту справку"
    Write-Host ""
    Write-Host "Примеры:"
    Write-Host "  .\run-worker.ps1                        # Стандартный запуск"
    Write-Host "  .\run-worker.ps1 -LogLevel debug        # Детальное логирование"
    Write-Host "  .\run-worker.ps1 -Queue ml              # Только ML задачи"
    Write-Host "  .\run-worker.ps1 -Beat                  # С планировщиком"
    Write-Host "  .\run-worker.ps1 -Flower                # С веб-интерфейсом"
    exit 0
}

Write-Host "⚙️ Запуск DiagMod Worker..." -ForegroundColor Green
Write-Host "🔧 Уровень логирования: $LogLevel" -ForegroundColor Yellow
Write-Host "👥 Параллельность: $Concurrency" -ForegroundColor Yellow

if ($Queue) {
    Write-Host "📋 Очередь: $Queue" -ForegroundColor Yellow
}

# Проверяем активацию виртуального окружения
if (-not $env:VIRTUAL_ENV) {
    Write-Warning "Виртуальное окружение не активировано. Активируем автоматически..."
    if (Test-Path "venv\Scripts\Activate.ps1") {
        & "venv\Scripts\Activate.ps1"
        Write-Host "✅ Виртуальное окружение активировано" -ForegroundColor Green
    } else {
        Write-Error "Виртуальное окружение не найдено. Запустите setup.ps1"
        exit 1
    }
}

# Запуск HTTP сервера метрик Worker в фоне
Write-Host "📊 Запуск сервера метрик на порту 8002..." -ForegroundColor Cyan
Start-Process -NoNewWindow -FilePath "python" -ArgumentList "scripts\worker_metrics_server.py"

try {
    if ($Beat) {
        Write-Host "📅 Запуск с планировщиком задач (Beat)..." -ForegroundColor Cyan
        celery -A src.worker.celery_app worker --beat --loglevel=$LogLevel --concurrency=$Concurrency
    } elseif ($Flower) {
        Write-Host "🌸 Запуск с Flower мониторингом..." -ForegroundColor Cyan
        Write-Host "🌐 Flower интерфейс: http://localhost:5555" -ForegroundColor Yellow

        # Запускаем Worker в фоне
        Start-Process -NoNewWindow -FilePath "celery" -ArgumentList "-A", "src.worker.celery_app", "worker", "--loglevel=$LogLevel", "--concurrency=$Concurrency"

        # Запускаем Flower
        celery -A src.worker.celery_app flower
    } elseif ($Queue) {
        Write-Host "📋 Обработка очереди: $Queue" -ForegroundColor Cyan
        celery -A src.worker.celery_app worker --loglevel=$LogLevel --concurrency=$Concurrency --queues=$Queue
    } else {
        celery -A src.worker.celery_app worker --loglevel=$LogLevel --concurrency=$Concurrency
    }
} catch {
    Write-Error "Ошибка запуска Worker: $_"
    Write-Host "Убедитесь, что:"
    Write-Host "  1. Установлены все зависимости (.\setup.ps1 -Dev)"
    Write-Host "  2. Запущен Redis (docker-compose up -d redis)"
    Write-Host "  3. База данных доступна"
    exit 1
}
