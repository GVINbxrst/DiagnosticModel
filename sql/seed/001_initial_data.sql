-- =============================================================================
-- DiagMod Database Seed Data - Initial System Data
-- Начальные данные для инициализации системы
-- =============================================================================

SET search_path TO diagmod, public;

-- =============================================================================
-- НАЧАЛЬНАЯ КОНФИГУРАЦИЯ СИСТЕМЫ
-- =============================================================================

INSERT INTO system_config (key, value, description, is_sensitive) VALUES
-- Общие настройки
('system.name', '"DiagMod"', 'Название системы', false),
('system.version', '"1.0.0"', 'Версия системы', false),
('system.timezone', '"UTC"', 'Часовой пояс системы', false),

-- Настройки обработки данных
('data.max_file_size_mb', '100', 'Максимальный размер CSV файла в МБ', false),
('data.sample_rate_default', '1000', 'Частота дискретизации по умолчанию (Гц)', false),
('data.window_size_ms', '1024', 'Размер окна для анализа в миллисекундах', false),
('data.overlap_ratio', '0.5', 'Коэффициент перекрытия окон', false),
('data.retention_days', '365', 'Период хранения данных в днях', false),

-- Настройки ML моделей
('ml.anomaly_threshold', '0.7', 'Порог для обнаружения аномалий', false),
('ml.retrain_interval_hours', '24', 'Интервал переобучения моделей в часах', false),
('ml.min_samples_training', '1000', 'Минимальное количество образцов для обучения', false),
('ml.feature_extraction_workers', '4', 'Количество процессов для извлечения признаков', false),

-- Настройки уведомлений
('alerts.enabled', 'true', 'Включены ли уведомления', false),
('alerts.critical_threshold', '0.9', 'Порог критических уведомлений', false),
('alerts.email_enabled', 'false', 'Отправка уведомлений по email', false),
('alerts.max_per_hour', '10', 'Максимальное количество уведомлений в час', false),

-- Настройки производительности
('performance.batch_size', '1000', 'Размер batch для обработки данных', false),
('performance.max_concurrent_jobs', '8', 'Максимальное количество параллельных задач', false),
('performance.cache_ttl_minutes', '60', 'Время жизни кеша в минутах', false),

-- Настройки безопасности
('security.jwt_expiry_minutes', '30', 'Время жизни JWT токена в минутах', false),
('security.max_login_attempts', '5', 'Максимальное количество попыток входа', false),
('security.session_timeout_hours', '8', 'Таймаут сессии в часах', false),
('security.password_min_length', '8', 'Минимальная длина пароля', false);

-- =============================================================================
-- ПОЛЬЗОВАТЕЛИ ПО УМОЛЧАНИЮ
-- =============================================================================

-- Администратор системы
INSERT INTO users (
    username,
    email,
    password_hash,
    full_name,
    role,
    is_active
) VALUES (
    'admin',
    'admin@diagmod.com',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewmVx8eZvfKDbfWy', -- password: admin123
    'Системный администратор',
    'admin',
    true
), (
    'engineer',
    'engineer@diagmod.com',
    '$2b$12$92IXUNpkjO0rOQ5byMi.Ye4oKoEa3Ro9llC/.og/at2.uheWG/igi', -- password: engineer123
    'Главный инженер',
    'engineer',
    true
), (
    'operator',
    'operator@diagmod.com',
    '$2b$12$ZQJ8f6x.M8E6iU7j5Y9G2OXnG.8v7s6s9T3k.X5L8a9c6h4s8f2', -- password: operator123
    'Оператор системы',
    'operator',
    true
);

-- =============================================================================
-- ТИПЫ ДЕФЕКТОВ
-- =============================================================================

INSERT INTO defect_types (code, name, description, category, default_severity) VALUES
-- Дефекты подшипников
('BEAR_OUTER', 'Дефект наружного кольца подшипника', 'Повреждение наружного кольца подшипника, обычно проявляется как повышенная вибрация на характерных частотах', 'bearing', 'medium'),
('BEAR_INNER', 'Дефект внутреннего кольца подшипника', 'Повреждение внутреннего кольца подшипника, критично для работы двигателя', 'bearing', 'high'),
('BEAR_BALL', 'Дефект тел качения подшипника', 'Повреждение шариков или роликов подшипника', 'bearing', 'medium'),
('BEAR_CAGE', 'Дефект сепаратора подшипника', 'Повреждение или износ сепаратора подшипника', 'bearing', 'medium'),
('BEAR_WEAR', 'Общий износ подшипника', 'Равномерный износ компонентов подшипника', 'bearing', 'low'),

-- Дефекты ротора
('ROTOR_UNBAL', 'Дисбаланс ротора', 'Неравномерное распределение массы ротора, приводящее к повышенной вибрации', 'rotor', 'medium'),
('ROTOR_MISALIGN', 'Перекос ротора', 'Нарушение соосности ротора и статора', 'rotor', 'medium'),
('ROTOR_BROKEN_BAR', 'Обрыв стержня ротора', 'Обрыв или трещина в стержне беличьей клетки', 'rotor', 'high'),
('ROTOR_ECCENTRICITY', 'Эксцентриситет ротора', 'Смещение оси ротора относительно геометрического центра', 'rotor', 'medium'),
('ROTOR_THERMAL', 'Термические деформации ротора', 'Деформации ротора из-за неравномерного нагрева', 'rotor', 'medium'),

-- Дефекты статора
('STATOR_WINDING', 'Дефект обмотки статора', 'Повреждение изоляции или обрыв витков обмотки статора', 'stator', 'high'),
('STATOR_CORE', 'Дефект магнитопровода статора', 'Повреждение пластин магнитопровода или их замыкание', 'stator', 'high'),
('STATOR_SLOT', 'Дефект пазов статора', 'Повреждение пазовой изоляции или деформация пазов', 'stator', 'medium'),

-- Электрические дефекты
('ELEC_PHASE_IMBALANCE', 'Дисбаланс фаз', 'Неравномерность токов или напряжений по фазам', 'electrical', 'medium'),
('ELEC_INSULATION', 'Снижение сопротивления изоляции', 'Ухудшение изоляционных свойств обмоток', 'electrical', 'high'),
('ELEC_PARTIAL_DISCHARGE', 'Частичные разряды', 'Локальные пробои изоляции в областях высокой напряженности поля', 'electrical', 'high'),
('ELEC_OVERHEAT', 'Перегрев обмоток', 'Превышение допустимой температуры обмоток', 'electrical', 'medium'),

-- Механические дефекты
('MECH_COUPLING', 'Дефект муфты', 'Повреждение или износ соединительной муфты', 'mechanical', 'medium'),
('MECH_FOUNDATION', 'Дефект фундамента', 'Нарушение жесткости или целостности фундамента', 'mechanical', 'low'),
('MECH_VIBRATION', 'Повышенная вибрация', 'Общая повышенная вибрация без явной причины', 'mechanical', 'low'),
('MECH_NOISE', 'Повышенный шум', 'Нехарактерные шумы при работе двигателя', 'mechanical', 'low'),

-- Дефекты системы охлаждения
('COOL_FAN', 'Дефект вентилятора охлаждения', 'Повреждение или износ лопастей вентилятора', 'cooling', 'medium'),
('COOL_AIRFLOW', 'Нарушение воздушного потока', 'Засорение или блокировка системы охлаждения', 'cooling', 'low'),

-- Прочие дефекты
('OTHER_LOAD', 'Перегрузка двигателя', 'Работа двигателя с нагрузкой выше номинальной', 'other', 'medium'),
('OTHER_SPEED', 'Нарушение скоростного режима', 'Отклонение скорости вращения от номинальной', 'other', 'low'),
('OTHER_UNKNOWN', 'Неопределенная аномалия', 'Обнаружена аномалия неясной природы', 'other', 'low');

-- =============================================================================
-- ТЕСТОВОЕ ОБОРУДОВАНИЕ
-- =============================================================================

INSERT INTO equipment (
    equipment_id,
    name,
    type,
    status,
    manufacturer,
    model,
    serial_number,
    installation_date,
    location,
    specifications
) VALUES
(
    'EQ_2025_000001',
    'Насос центробежный НЦ-320',
    'pump',
    'active',
    'Grundfos',
    'NK 65-125/124',
    'GF2024001',
    '2024-01-15',
    'Цех №1, Участок водоснабжения',
    jsonb_build_object(
        'power_kw', 15,
        'voltage_v', 380,
        'current_a', 28.5,
        'frequency_hz', 50,
        'rpm', 1450,
        'efficiency', 0.85,
        'protection_class', 'IP55',
        'bearing_de', '6309',
        'bearing_nde', '6207'
    )
),
(
    'EQ_2025_000002',
    'Двигатель главного привода М1',
    'induction_motor',
    'active',
    'ABB',
    'M3BP 132 SMB 4',
    'ABB2024002',
    '2024-02-20',
    'Цех №2, Линия производства',
    jsonb_build_object(
        'power_kw', 7.5,
        'voltage_v', 380,
        'current_a', 14.2,
        'frequency_hz', 50,
        'rpm', 1440,
        'efficiency', 0.88,
        'protection_class', 'IP55',
        'bearing_de', '6206',
        'bearing_nde', '6204'
    )
),
(
    'EQ_2025_000003',
    'Вентилятор вытяжной В-01',
    'fan',
    'active',
    'Systemair',
    'DVNI 315D4-8',
    'SYS2024003',
    '2024-03-10',
    'Цех №1, Система вентиляции',
    jsonb_build_object(
        'power_kw', 3.0,
        'voltage_v', 380,
        'current_a', 6.1,
        'frequency_hz', 50,
        'rpm', 720,
        'efficiency', 0.82,
        'protection_class', 'IP54'
    )
),
(
    'EQ_2025_000004',
    'Компрессор винтовой К-150',
    'compressor',
    'maintenance',
    'Atlas Copco',
    'GA 15 VSD+',
    'AC2024004',
    '2024-01-30',
    'Компрессорная станция',
    jsonb_build_object(
        'power_kw', 15,
        'voltage_v', 380,
        'current_a', 27.8,
        'frequency_hz', 50,
        'rpm', 1800,
        'efficiency', 0.90,
        'protection_class', 'IP55',
        'pressure_bar', 8
    )
),
(
    'EQ_2025_000005',
    'Конвейер ленточный КЛ-500',
    'conveyor',
    'active',
    'Siemens',
    '1LA7 113-4AA10',
    'SIE2024005',
    '2024-04-05',
    'Цех №3, Транспортная линия',
    jsonb_build_object(
        'power_kw', 4.0,
        'voltage_v', 380,
        'current_a', 8.5,
        'frequency_hz', 50,
        'rpm', 1450,
        'efficiency', 0.85,
        'protection_class', 'IP55',
        'belt_speed_ms', 1.2
    )
);

-- =============================================================================
-- СОБЫТИЯ ОБСЛУЖИВАНИЯ
-- =============================================================================

INSERT INTO maintenance_events (
    equipment_id,
    user_id,
    event_type,
    status,
    scheduled_date,
    started_at,
    completed_at,
    description,
    cost,
    parts_replaced,
    notes
) VALUES
-- Завершенное плановое обслуживание
(
    (SELECT id FROM equipment WHERE equipment_id = 'EQ_2025_000001'),
    (SELECT id FROM users WHERE username = 'engineer'),
    'planned',
    'completed',
    '2025-01-05 10:00:00+00',
    '2025-01-05 10:15:00+00',
    '2025-01-05 14:30:00+00',
    'Плановое техническое обслуживание. Замена подшипников, проверка обмоток.',
    25000.00,
    ARRAY['Подшипник 6309', 'Подшипник 6207', 'Смазка Shell Gadus S2'],
    'Обнаружен повышенный износ подшипника DE. Произведена замена.'
),

-- Запланированное обслуживание
(
    (SELECT id FROM equipment WHERE equipment_id = 'EQ_2025_000002'),
    (SELECT id FROM users WHERE username = 'engineer'),
    'planned',
    'scheduled',
    '2025-01-15 09:00:00+00',
    NULL,
    NULL,
    'Плановая проверка и диагностика. Измерение сопротивления изоляции.',
    15000.00,
    NULL,
    NULL
),

-- Экстренное обслуживание в процессе
(
    (SELECT id FROM equipment WHERE equipment_id = 'EQ_2025_000004'),
    (SELECT id FROM users WHERE username = 'engineer'),
    'emergency',
    'in_progress',
    '2025-01-08 08:00:00+00',
    '2025-01-08 08:30:00+00',
    NULL,
    'Экстренная остановка из-за повышенной вибрации. Диагностика подшипников.',
    NULL,
    NULL,
    'Обнаружена повышенная вибрация на частоте BPFO. Требуется замена подшипника DE.'
);

-- =============================================================================
-- СИСТЕМНЫЕ ЛОГИ ДЛЯ ДЕМОНСТРАЦИИ
-- =============================================================================

INSERT INTO system_logs (level, module, message, details, user_id) VALUES
('INFO', 'system', 'Система DiagMod успешно инициализирована',
 jsonb_build_object('version', '1.0.0', 'timestamp', CURRENT_TIMESTAMP),
 (SELECT id FROM users WHERE username = 'admin')),

('INFO', 'database', 'Начальные данные загружены',
 jsonb_build_object(
     'users_count', (SELECT COUNT(*) FROM users),
     'equipment_count', (SELECT COUNT(*) FROM equipment),
     'defect_types_count', (SELECT COUNT(*) FROM defect_types)
 ),
 (SELECT id FROM users WHERE username = 'admin')),

('INFO', 'config', 'Конфигурация системы установлена',
 jsonb_build_object('config_entries', (SELECT COUNT(*) FROM system_config)),
 (SELECT id FROM users WHERE username = 'admin'));

-- =============================================================================
-- ОБНОВЛЕНИЕ ПОСЛЕДОВАТЕЛЬНОСТЕЙ
-- =============================================================================

-- Установка следующего значения для equipment_id_seq
SELECT setval('equipment_id_seq', 5);

-- =============================================================================
-- ФИНАЛЬНАЯ ПРОВЕРКА И СТАТИСТИКА
-- =============================================================================

-- Обновление статистики для оптимизатора
ANALYZE users;
ANALYZE equipment;
ANALYZE defect_types;
ANALYZE maintenance_events;
ANALYZE system_config;
ANALYZE system_logs;

-- Вывод сводной информации о загруженных данных
DO $$
DECLARE
    users_count INTEGER;
    equipment_count INTEGER;
    defect_types_count INTEGER;
    maintenance_count INTEGER;
    config_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO users_count FROM users;
    SELECT COUNT(*) INTO equipment_count FROM equipment;
    SELECT COUNT(*) INTO defect_types_count FROM defect_types;
    SELECT COUNT(*) INTO maintenance_count FROM maintenance_events;
    SELECT COUNT(*) INTO config_count FROM system_config;

    RAISE NOTICE '=== DiagMod Database Initialization Complete ===';
    RAISE NOTICE 'Users: %', users_count;
    RAISE NOTICE 'Equipment: %', equipment_count;
    RAISE NOTICE 'Defect Types: %', defect_types_count;
    RAISE NOTICE 'Maintenance Events: %', maintenance_count;
    RAISE NOTICE 'System Config: %', config_count;
    RAISE NOTICE 'Database ready for operation!';
END $$;
