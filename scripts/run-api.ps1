# DiagMod API Server - PowerShell Launch Script
# Запуск FastAPI сервера в режиме разработки

param(
    [string]$Host = "0.0.0.0",
    [int]$Port = 8000,
    [switch]$Reload = $true,
    [switch]$Help
)

if ($Help) {
    Write-Host "DiagMod API Server" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Параметры:"
    Write-Host "  -Host <ip>     IP адрес для привязки (по умолчанию: 0.0.0.0)"
    Write-Host "  -Port <port>   Порт для прослушивания (по умолчанию: 8000)"
    Write-Host "  -Reload        Автоперезагрузка при изменениях (по умолчанию: включена)"
    Write-Host "  -Help          Показать эту справку"
    Write-Host ""
    Write-Host "Примеры:"
    Write-Host "  .\run-api.ps1                    # Стандартный запуск"
    Write-Host "  .\run-api.ps1 -Port 8080         # Запуск на порту 8080"
    Write-Host "  .\run-api.ps1 -Host 127.0.0.1    # Только локальный доступ"
    exit 0
}

Write-Host "🚀 Запуск DiagMod API Server..." -ForegroundColor Green
Write-Host "📡 Сервер будет доступен по адресу: http://$Host`:$Port" -ForegroundColor Yellow
Write-Host "📚 API документация: http://$Host`:$Port/docs" -ForegroundColor Yellow
Write-Host "📊 Метрики: http://$Host`:$Port/metrics" -ForegroundColor Yellow
Write-Host ""

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

# Запускаем API сервер
try {
    if ($Reload) {
        uvicorn src.api.main:app --reload --host $Host --port $Port
    } else {
        uvicorn src.api.main:app --host $Host --port $Port
    }
} catch {
    Write-Error "Ошибка запуска API сервера: $_"
    Write-Host "Убедитесь, что:"
    Write-Host "  1. Установлены все зависимости (.\setup.ps1 -Dev)"
    Write-Host "  2. Запущена база данных (docker-compose up -d postgres redis)"
    exit 1
}
