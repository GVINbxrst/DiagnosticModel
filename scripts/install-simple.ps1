# DiagMod - Упрощенная установка зависимостей для Windows
# Этот скрипт устанавливает зависимости пошагово с диагностикой

# Настройка кодировки для корректного отображения русского текста
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 | Out-Null

Write-Host "🔍 Диагностика системы DiagMod..." -ForegroundColor Cyan
Write-Host ""

# Проверка Python
Write-Host "1. Проверка Python..." -ForegroundColor Yellow
try {
    $pythonVersion = & python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Python найден: $pythonVersion" -ForegroundColor Green
        $pythonCmd = "python"
    } else {
        $pythonVersion = & py --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Python найден через py launcher: $pythonVersion" -ForegroundColor Green
            $pythonCmd = "py"
        } else {
            Write-Host "❌ Python не ��айден в PATH" -ForegroundColor Red
            Write-Host "Установите Python с https://python.org и добавьте в PATH" -ForegroundColor Yellow
            exit 1
        }
    }
} catch {
    Write-Host "❌ Ошибка проверки Python: $_" -ForegroundColor Red
    exit 1
}

# Проверка pip
Write-Host "`n2. Проверка pip..." -ForegroundColor Yellow
try {
    $pipVersion = & $pythonCmd -m pip --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ pip найден: $pipVersion" -ForegroundColor Green
    } else {
        Write-Host "❌ pip не найден" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "❌ Ошибка проверки pip: $_" -ForegroundColor Red
    exit 1
}

# Проверка виртуального окружения
Write-Host "`n3. Проверка виртуального окружения..." -ForegroundColor Yellow
if ($env:VIRTUAL_ENV) {
    Write-Host "✅ Виртуальное окружение активно: $env:VIRTUAL_ENV" -ForegroundColor Green
} else {
    Write-Host "⚠️ Виртуальное окружение не активно" -ForegroundColor Yellow

    if (Test-Path "venv") {
        Write-Host "Найдено существующее виртуальное окружение. Активируем..." -ForegroundColor Yellow
        try {
            & "venv\Scripts\Activate.ps1"
            Write-Host "✅ Виртуальное окружение активировано" -ForegroundColor Green
        } catch {
            Write-Host "❌ Ошибка активации: $_" -ForegroundColor Red
            Write-Host "Создаем новое виртуальное окружение..." -ForegroundColor Yellow
            & $pythonCmd -m venv venv
            & "venv\Scripts\Activate.ps1"
        }
    } else {
        Write-Host "Создаем виртуальное окружение..." -ForegroundColor Yellow
        & $pythonCmd -m venv venv
        & "venv\Scripts\Activate.ps1"
        Write-Host "✅ Виртуальное окружение создано и активировано" -ForegroundColor Green
    }
}

# Обновление pip
Write-Host "`n4. Обновление pip..." -ForegroundColor Yellow
try {
    & $pythonCmd -m pip install --upgrade pip setuptools wheel
    Write-Host "✅ pip обновлен" -ForegroundColor Green
} catch {
    Write-Host "❌ Ошибка обновления pip: $_" -ForegroundColor Red
}

# Установка критически важных пакетов
Write-Host "`n5. Установка базовых пакетов..." -ForegroundColor Yellow

$basicPackages = @(
    "fastapi",
    "uvicorn[standard]",
    "streamlit",
    "pydantic",
    "requests",
    "python-dotenv"
)

foreach ($package in $basicPackages) {
    Write-Host "   Устанавливаем $package..." -ForegroundColor Cyan
    try {
        & $pythonCmd -m pip install $package --quiet
        Write-Host "   ✅ $package установлен" -ForegroundColor Green
    } catch {
        Write-Host "   ❌ Ошибка установки $package" -ForegroundColor Red
    }
}

Write-Host "`n6. Установка пакетов для работы с данными..." -ForegroundColor Yellow

$dataPackages = @(
    "numpy",
    "pandas",
    "matplotlib",
    "plotly"
)

foreach ($package in $dataPackages) {
    Write-Host "   Устанавливаем $package..." -ForegroundColor Cyan
    try {
        & $pythonCmd -m pip install $package --quiet
        Write-Host "   ✅ $package установлен" -ForegroundColor Green
    } catch {
        Write-Host "   ❌ Ошибка установки $package" -ForegroundColor Red
    }
}

Write-Host "`n7. Установка ML библиотек..." -ForegroundColor Yellow

$mlPackages = @(
    "scikit-learn",
    "joblib"
)

foreach ($package in $mlPackages) {
    Write-Host "   Устанавливаем $package..." -ForegroundColor Cyan
    try {
        & $pythonCmd -m pip install $package --quiet
        Write-Host "   ✅ $package установлен" -ForegroundColor Green
    } catch {
        Write-Host "   ❌ Ошибка установки $package" -ForegroundColor Red
    }
}

Write-Host "`n8. Проверка установленных пакетов..." -ForegroundColor Yellow
try {
    & $pythonCmd -c "import fastapi, streamlit, numpy, pandas; print('✅ Базовые пакеты работают!')"
    Write-Host "✅ Базовая установка завершена успешно!" -ForegroundColor Green
} catch {
    Write-Host "❌ Ошибка проверки пакетов: $_" -ForegroundColor Red
}

Write-Host "`n9. Установка дополнительных пакетов..." -ForegroundColor Yellow

# Установка только совместимых пакетов
$additionalPackages = @(
    "sqlalchemy",
    "asyncpg",
    "alembic",
    "celery[redis]",
    "redis",
    "prometheus-client",
    "python-jose[cryptography]",
    "passlib[bcrypt]",
    "reportlab"
)

foreach ($package in $additionalPackages) {
    Write-Host "   Устанавливаем $package..." -ForegroundColor Cyan
    try {
        & $pythonCmd -m pip install $package --quiet
        Write-Host "   ✅ $package установлен" -ForegroundColor Green
    } catch {
        Write-Host "   ⚠️ Пропускаем $package (проблемы совместимости)" -ForegroundColor Yellow
    }
}

Write-Host "`n✅ Установка завершена!" -ForegroundColor Green
Write-Host "`n📋 Следующие шаги:" -ForegroundColor Cyan
Write-Host "1. Настройка базы данных:" -ForegroundColor White
Write-Host "   docker-compose up -d postgres redis" -ForegroundColor Gray
Write-Host "2. Инициализация схемы БД:" -ForegroundColor White
Write-Host "   python -m alembic upgrade head" -ForegroundColor Gray
Write-Host "3. Запуск компонентов:" -ForegroundColor White
Write-Host "   .\scripts\run-api.ps1        # FastAPI сервер" -ForegroundColor Gray
Write-Host "   .\scripts\run-dashboard.ps1  # Streamlit Dashboard" -ForegroundColor Gray
