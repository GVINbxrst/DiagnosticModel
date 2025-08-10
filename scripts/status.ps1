# DiagMod System Status Check - PowerShell Script
# Проверка статуса всех компонентов системы

param(
    [switch]$Detailed,
    [switch]$Json,
    [switch]$Help
)

if ($Help) {
    Write-Host "DiagMod System Status Checker" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Параметры:"
    Write-Host "  -Detailed    Детальная проверка всех компонентов"
    Write-Host "  -Json        Вывод в формате JSON"
    Write-Host "  -Help        Показать эту справку"
    Write-Host ""
    Write-Host "Примеры:"
    Write-Host "  .\status.ps1                # Быстрая проверка"
    Write-Host "  .\status.ps1 -Detailed      # Полная диагностика"
    Write-Host "  .\status.ps1 -Json          # JSON вывод"
    exit 0
}

# Функции для проверки компонентов
function Test-ServiceHealth {
    param($Url, $ServiceName, $TimeoutSeconds = 5)

    try {
        $response = Invoke-WebRequest -Uri $Url -TimeoutSec $TimeoutSeconds -UseBasicParsing
        return @{
            Name = $ServiceName
            Status = "Healthy"
            StatusCode = $response.StatusCode
            ResponseTime = $response.Headers.'Response-Time'
        }
    }
    catch {
        return @{
            Name = $ServiceName
            Status = "Unhealthy"
            Error = $_.Exception.Message
        }
    }
}

function Test-DockerService {
    param($ServiceName)

    try {
        $output = docker-compose ps $ServiceName 2>$null
        if ($LASTEXITCODE -eq 0 -and $output -match "Up") {
            return @{
                Name = $ServiceName
                Status = "Running"
                Container = "Up"
            }
        } else {
            return @{
                Name = $ServiceName
                Status = "Stopped"
                Container = "Down"
            }
        }
    }
    catch {
        return @{
            Name = $ServiceName
            Status = "Unknown"
            Error = "Docker not available"
        }
    }
}

# Главная функция проверки
function Get-SystemStatus {
    $status = @{
        Timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
        OverallStatus = "Unknown"
        Services = @{}
        Docker = @{}
        System = @{}
    }

    Write-Host "🔍 Проверка статуса DiagMod системы..." -ForegroundColor Cyan
    Write-Host ""

    # Проверка веб-сервисов
    Write-Host "🌐 Проверка веб-сервисов..." -ForegroundColor Yellow

    $webServices = @(
        @{ Url = "http://localhost:8000/health"; Name = "API Server" },
        @{ Url = "http://localhost:8501"; Name = "Dashboard" },
        @{ Url = "http://localhost:9090"; Name = "Prometheus" },
        @{ Url = "http://localhost:8000/metrics"; Name = "API Metrics" },
        @{ Url = "http://localhost:8002/health"; Name = "Worker Metrics" }
    )

    foreach ($service in $webServices) {
        $result = Test-ServiceHealth -Url $service.Url -ServiceName $service.Name
        $status.Services[$service.Name] = $result

        if ($result.Status -eq "Healthy") {
            Write-Host "  ✅ $($service.Name)" -ForegroundColor Green
        } else {
            Write-Host "  ❌ $($service.Name) - $($result.Error)" -ForegroundColor Red
        }
    }

    Write-Host ""

    # Проверка Docker контейнеров
    Write-Host "🐳 Проверка Docker контейнеров..." -ForegroundColor Yellow

    $dockerServices = @("postgres", "redis", "api", "worker", "dashboard", "prometheus")

    foreach ($service in $dockerServices) {
        $result = Test-DockerService -ServiceName $service
        $status.Docker[$service] = $result

        if ($result.Status -eq "Running") {
            Write-Host "  ✅ $service" -ForegroundColor Green
        } elseif ($result.Status -eq "Stopped") {
            Write-Host "  ⏹️ $service (остановлен)" -ForegroundColor Yellow
        } else {
            Write-Host "  ❓ $service (неизвестно)" -ForegroundColor Gray
        }
    }

    Write-Host ""

    # Системная информация
    if ($Detailed) {
        Write-Host "💻 Системная информация..." -ForegroundColor Yellow

        # Процессы Python
        $pythonProcesses = Get-Process python* -ErrorAction SilentlyContinue
        $status.System.PythonProcesses = $pythonProcesses.Count

        # Использование портов
        $usedPorts = @()
        $checkPorts = @(8000, 8501, 8502, 9090, 5432, 6379)

        foreach ($port in $checkPorts) {
            $connection = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
            if ($connection) {
                $usedPorts += $port
            }
        }
        $status.System.UsedPorts = $usedPorts

        Write-Host "  📊 Python процессов: $($pythonProcesses.Count)" -ForegroundColor Cyan
        Write-Host "  🔌 Занятых портов: $($usedPorts -join ', ')" -ForegroundColor Cyan

        # Статус виртуального окружения
        if ($env:VIRTUAL_ENV) {
            Write-Host "  🐍 Виртуальное окружение: активно ($env:VIRTUAL_ENV)" -ForegroundColor Green
            $status.System.VirtualEnv = "Active"
        } else {
            Write-Host "  🐍 Виртуальное окружение: не активно" -ForegroundColor Yellow
            $status.System.VirtualEnv = "Inactive"
        }

        Write-Host ""
    }

    # Определение общего статуса
    $healthyServices = ($status.Services.Values | Where-Object { $_.Status -eq "Healthy" }).Count
    $totalServices = $status.Services.Count
    $runningContainers = ($status.Docker.Values | Where-Object { $_.Status -eq "Running" }).Count

    if ($healthyServices -eq $totalServices -and $runningContainers -gt 0) {
        $status.OverallStatus = "Healthy"
        Write-Host "🟢 Общий статус: Система работает нормально" -ForegroundColor Green
    } elseif ($healthyServices -gt 0 -or $runningContainers -gt 0) {
        $status.OverallStatus = "Degraded"
        Write-Host "🟡 Общий статус: Система работает частично" -ForegroundColor Yellow
    } else {
        $status.OverallStatus = "Unhealthy"
        Write-Host "🔴 Общий статус: Система не работает" -ForegroundColor Red
    }

    Write-Host ""
    Write-Host "📈 Сводка:" -ForegroundColor Cyan
    Write-Host "  Веб-сервисов активно: $healthyServices/$totalServices" -ForegroundColor White
    Write-Host "  Docker контейнеров запущено: $runningContainers/$($dockerServices.Count)" -ForegroundColor White

    return $status
}

# Запуск проверки
$systemStatus = Get-SystemStatus

# Вывод в JSON если запрошен
if ($Json) {
    $systemStatus | ConvertTo-Json -Depth 3
}

# Рекомендации по устранению проблем
if ($systemStatus.OverallStatus -ne "Healthy") {
    Write-Host ""
    Write-Host "🔧 Рекомендации по устранению проблем:" -ForegroundColor Yellow
    Write-Host ""

    $healthyServices = ($systemStatus.Services.Values | Where-Object { $_.Status -eq "Healthy" }).Count
    $runningContainers = ($systemStatus.Docker.Values | Where-Object { $_.Status -eq "Running" }).Count

    if ($runningContainers -eq 0) {
        Write-Host "  1. Запустите Docker сервисы:" -ForegroundColor White
        Write-Host "     docker-compose up -d" -ForegroundColor Gray
        Write-Host ""
    }

    if ($healthyServices -eq 0) {
        Write-Host "  2. Проверьте логи сервисов:" -ForegroundColor White
        Write-Host "     docker-compose logs api" -ForegroundColor Gray
        Write-Host "     docker-compose logs worker" -ForegroundColor Gray
        Write-Host ""

        Write-Host "  3. Перезапустите проблемные сервисы:" -ForegroundColor White
        Write-Host "     docker-compose restart api worker" -ForegroundColor Gray
        Write-Host ""
    }

    if (-not $env:VIRTUAL_ENV) {
        Write-Host "  4. Активируйте виртуальное окружение:" -ForegroundColor White
        Write-Host "     venv\Scripts\Activate.ps1" -ForegroundColor Gray
        Write-Host ""
    }

    Write-Host "  5. Полная переустановка (если ничего не помогает):" -ForegroundColor White
    Write-Host "     .\scripts\setup.ps1 -Dev" -ForegroundColor Gray
}

Write-Host ""
Write-Host "ℹ️ Для получения помощи используйте: .\status.ps1 -Help" -ForegroundColor Blue
