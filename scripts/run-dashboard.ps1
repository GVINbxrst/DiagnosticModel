# DiagMod Dashboard - PowerShell Launch Script
# Запуск Streamlit Dashboard

param(
    [string]$Host = "0.0.0.0",
    [int]$Port = 8501,
    [string]$ApiUrl = "http://localhost:8000",
    [switch]$Help
)

if ($Help) {
    Write-Host "DiagMod Streamlit Dashboard" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Параметры:"
    Write-Host "  -Host <ip>        IP адрес для привязки (по умолчанию: 0.0.0.0)"
    Write-Host "  -Port <port>      Порт для прослушивания (по умолчанию: 8501)"
    Write-Host "  -ApiUrl <url>     URL API сервера (по умолчанию: http://localhost:8000)"
    Write-Host "  -Help             Показать эту справку"
    Write-Host ""
    Write-Host "Примеры:"
    Write-Host "  .\run-dashboard.ps1                              # Стандартный запуск"
    Write-Host "  .\run-dashboard.ps1 -Port 8080                   # Другой порт"
    Write-Host "  .\run-dashboard.ps1 -ApiUrl http://api:8000      # Внешний API"
    exit 0
}

Write-Host "📊 Запуск DiagMod Dashboard..." -ForegroundColor Green
Write-Host "🌐 Dashboard будет доступен: http://$Host`:$Port" -ForegroundColor Yellow
Write-Host "🔗 API сервер: $ApiUrl" -ForegroundColor Yellow
Write-Host ""

# Устанавливаем переменные окружения для Dashboard
$env:API_BASE_URL = $ApiUrl
$env:DASHBOARD_HOST = $Host
$env:DASHBOARD_PORT = $Port

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

# Проверяем доступность API
Write-Host "🔍 Проверка доступности API..." -ForegroundColor Cyan
try {
    $response = Invoke-WebRequest -Uri "$ApiUrl/health" -TimeoutSec 5 -UseBasicParsing
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ API сервер доступен" -ForegroundColor Green
    }
} catch {
    Write-Warning "API сервер недоступен по адресу $ApiUrl"
    Write-Host "Убедитесь, что API запущен: .\run-api.ps1" -ForegroundColor Yellow
}

# Запускаем Dashboard
try {
    streamlit run src/dashboard/main.py --server.port $Port --server.address $Host --server.headless true --browser.gatherUsageStats false
} catch {
    Write-Error "Ошибка запуска Dashboard: $_"
    Write-Host "Убедитесь, что:"
    Write-Host "  1. Установлены все зависимости (.\setup.ps1 -Dev)"
    Write-Host "  2. API сервер запущен (.\run-api.ps1)"
    Write-Host "  3. Порт $Port свободен"
    exit 1
}
