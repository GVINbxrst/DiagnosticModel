# Alembic Migrations

Инициализирован базовый каталог миграций. 

## Запуск автогенерации ревизии

Пример команды (PowerShell):

```
$env:DATABASE_URL="postgresql://user:pass@localhost:5432/diagmod"; \
python -m alembic revision --autogenerate -m "baseline"
```

Применение:
```
python -m alembic upgrade head
```

Настроен compare_type для обновлений типов.
