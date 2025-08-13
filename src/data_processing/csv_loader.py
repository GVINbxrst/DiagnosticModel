"""
CSV Loader для токовых сигналов асинхронных двигателей

Этот модуль обрабатывает большие CSV файлы с токовыми данными трех фаз:
- Читает файлы с одной колонкой данных
- Парсит заголовок вида "current_R,current_S,current_T"
- Обрабатывает пропущенные значения (NaN)
- Загружает данные пачками в PostgreSQL с gzip сжатием
- Логирует весь процесс обработки
"""

import asyncio
import csv
import hashlib
import os
import struct
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, AsyncGenerator
from uuid import UUID
import time

import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.settings import get_settings
from src.database.connection import get_async_session
from src.database.models import Equipment, RawSignal
from src.utils.logger import get_logger
from src.utils.serialization import dump_float32_array, load_float32_array
try:
    from src.utils import metrics as m  # type: ignore
except Exception:  # pragma: no cover - fallback for broken metrics during early steps
    class _DummyMetrics:  # minimal no-op interface
        @staticmethod
        def increment_counter(*args, **kwargs):
            return None
        @staticmethod
        def observe_histogram(*args, **kwargs):
            return None
    m = _DummyMetrics()  # type: ignore

# Настройки
settings = get_settings()
logger = get_logger(__name__)

# Константы
BATCH_SIZE = 10_000
DEFAULT_SAMPLE_RATE = 25_600  # Гц
PHASE_NAMES = ['R', 'S', 'T']
EXPECTED_HEADER_PATTERN = ['current_R', 'current_S', 'current_T']


class CSVLoaderError(Exception):
    """Базовое исключение для CSV Loader"""
    pass


class InvalidCSVFormatError(CSVLoaderError):
    """Исключение для некорректного формата CSV"""
    pass


class CSVProcessingStats:
    """Статистика обработки CSV файла.

    Исправлена проблема с потенциально повреждённой индентацией: полностью
    переопределён __init__ без смешения табов и пробелов.
    """

    def __init__(self) -> None:  # noqa: D401
        # Счетчики строк
        self.total_rows: int = 0
        self.processed_rows: int = 0
        self.skipped_rows: int = 0
        self.invalid_rows: int = 0
        # NaN по фазам
        self.nan_values: Dict[str, int] = {"R": 0, "S": 0, "T": 0}
        # Время
        self.start_time: datetime = datetime.now()
        self.end_time: Optional[datetime] = None
        # Пачки
        self.batches_processed: int = 0
        # Идентификаторы сохранённых RawSignal (может быть несколько)
        self.raw_signal_ids: List[UUID] = []
        # Общее количество семплов (максимум samples_count по пачкам)
        self.samples_count_total: int = 0
        # Какие фазы имели хотя бы одно ненулевое (не все NaN) значение
        self.phases_present: Dict[str, bool] = {"R": False, "S": False, "T": False}
        # Статус фаз (ok|missing) — заполняется в finish()
        self.phase_status: Dict[str, str] = {"R": "ok", "S": "ok", "T": "ok"}

    def add_batch_stats(
        self,
        batch_size: int,
        nan_counts: Dict[str, int],
        raw_id: Optional[UUID] = None,
        samples_count: int = 0,
        phases_present: Optional[Dict[str, bool]] = None,
    ):
        """Добавить статистику обработанной пачки"""
        self.processed_rows += batch_size
        self.total_rows += batch_size
        self.batches_processed += 1
        if raw_id:
            self.raw_signal_ids.append(raw_id)
        self.samples_count_total += samples_count
        if phases_present:
            for p, has in phases_present.items():
                if has:
                    self.phases_present[p] = True
        for phase, count in nan_counts.items():
            self.nan_values[phase] += count

    def finish(self):
        """Завершить подсчет статистики"""
        self.end_time = datetime.now()
        # Определяем статус фаз: если вся фаза пустая (все NaN) -> missing
        for p in ['R','S','T']:
            if self.processed_rows > 0 and self.nan_values.get(p, 0) >= self.processed_rows:
                self.phase_status[p] = 'missing'
            else:
                self.phase_status[p] = 'ok'

    @property
    def processing_time(self) -> float:
        """Время обработки в секундах"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.now() - self.start_time).total_seconds()

    @property
    def rows_per_second(self) -> float:
        """Скорость обработки строк в секунду"""
        time_elapsed = self.processing_time
        return self.processed_rows / time_elapsed if time_elapsed > 0 else 0

    def to_dict(self) -> Dict:
        """Конвертировать статистику в словарь"""
        return {
            'total_rows': self.total_rows,
            'processed_rows': self.processed_rows,
            'skipped_rows': self.skipped_rows,
            'invalid_rows': self.invalid_rows,
            'nan_values': self.nan_values,
            'processing_time_seconds': self.processing_time,
            'rows_per_second': self.rows_per_second,
            'batches_processed': self.batches_processed,
            'raw_signal_ids': [str(r) for r in self.raw_signal_ids],
            'samples_count_total': self.samples_count_total,
            'phases_present': self.phases_present,
            'phase_status': self.phase_status
        }

    # Совместимость со старым интерфейсом
    @property
    def raw_signal_id(self):  # type: ignore
        return self.raw_signal_ids[0] if self.raw_signal_ids else None

    @property
    def total_samples(self):  # type: ignore
        return self.samples_count_total


## Сжатие/распаковка теперь централизовано в src.utils.serialization.
## Оставляем совместимость: экспорт прежних имен для тестов, но перенаправляем.
def compress_float32_array(data: np.ndarray):  # type: ignore
    """Back-compat wrapper returning b'' for empty arrays to satisfy legacy tests."""
    b = dump_float32_array(data)
    return b if b is not None else b''


def decompress_float32_array(b: bytes):  # type: ignore
    arr = load_float32_array(b)
    return arr if arr is not None else np.array([], dtype=np.float32)


def calculate_file_hash(file_path: Path) -> str:
    """
    Вычислить SHA256 хеш файла для дедупликации

    Args:
        file_path: Путь к файлу

    Returns:
        SHA256 хеш в виде строки
    """
    hash_sha256 = hashlib.sha256()

    with open(file_path, 'rb') as f:
        # Читаем файл блоками для больших файлов
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def parse_csv_header(header_line: str) -> List[str]:
    """
    Парсить заголовок CSV файла

    Args:
        header_line: Первая строка файла

    Returns:
        Список названий фаз

    Raises:
        InvalidCSVFormatError: Если формат заголовка некорректный
    """
    # Убираем лишние пробелы и разбиваем по запятым
    phases = [phase.strip() for phase in header_line.strip().split(',')]

    # Проверяем количество фаз
    if len(phases) != 3:
        raise InvalidCSVFormatError(
            f"Ожидается 3 фазы, найдено {len(phases)}: {phases}"
        )

    # Проверяем формат названий фаз
    expected_patterns = ['current_R', 'current_S', 'current_T']
    for i, (expected, actual) in enumerate(zip(expected_patterns, phases)):
        if not actual.startswith('current_'):
            logger.warning(f"Нестандартное название фазы {i}: '{actual}', "
                          f"ожидается '{expected}'")

    logger.info(f"Обнаружены фазы: {phases}")
    return phases


def parse_csv_row(row_data: str, phase_count: int = 3) -> Tuple[List[float], List[bool]]:
    """
    Парсить строку данных CSV

    Args:
        row_data: Строка с данными фаз
        phase_count: Количество ожидаемых фаз

    Returns:
        Кортеж (значения фаз, маска NaN)
    """
    # Разбиваем строку по запятым
    values_str = [val.strip() for val in row_data.split(',')]

    # Дополняем до нужного количества фаз, если их меньше
    while len(values_str) < phase_count:
        values_str.append('')

    # Берем только первые phase_count значений
    values_str = values_str[:phase_count]

    values = []
    nan_mask = []

    for val_str in values_str:
        if val_str == '' or val_str.lower() in ['nan', 'null', 'none']:
            values.append(np.nan)
            nan_mask.append(True)
        else:
            try:
                values.append(float(val_str))
                nan_mask.append(False)
            except ValueError:
                logger.warning(f"Некорректное значение: '{val_str}', заменяем на NaN")
                values.append(np.nan)
                nan_mask.append(True)

    return values, nan_mask


async def find_equipment_by_filename(
    session: AsyncSession,
    filename: str
) -> Optional[Equipment]:
    """
    Найти оборудование по имени файла

    Args:
        session: Сессия базы данных
        filename: Имя файла

    Returns:
        Объект Equipment или None
    """
    from sqlalchemy import select

    # Пытаемся извлечь ID оборудования из имени файла
    # Ожидаемые форматы: motor_001.csv, EQ_2025_000001_data.csv и т.д.
    filename_lower = filename.lower()

    # Сначала ищем по точному совпадению в specifications
    result = await session.execute(
        select(Equipment).where(
            Equipment.specifications.op('->')('filename').as_string().ilike(f'%{filename}%')
        )
    )
    equipment = result.scalar_one_or_none()

    if equipment:
        return equipment

    # Если не найдено, ищем по equipment_id в имени файла
    if 'eq_' in filename_lower:
        # Извлекаем equipment_id из имени файла
        parts = filename_lower.split('_')
        if len(parts) >= 3:
            equipment_id = f"EQ_{parts[1]}_{parts[2].split('.')[0]}"

            result = await session.execute(
                select(Equipment).where(Equipment.equipment_id == equipment_id)
            )
            equipment = result.scalar_one_or_none()

            if equipment:
                return equipment

    # Если не найдено, возвращаем первое доступное оборудование
    result = await session.execute(
        select(Equipment).where(Equipment.status == 'active').limit(1)
    )
    equipment = result.scalar_one_or_none()

    if equipment:
        logger.warning(f"Оборудование для файла '{filename}' не найдено, "
                      f"используется {equipment.equipment_id}")

    return equipment


class CSVLoader:
    """Загрузчик CSV файлов с токовыми сигналами"""

    def __init__(self, batch_size: int = BATCH_SIZE):
        self.batch_size = batch_size
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

    async def load_csv_file(
        self,
        file_path: Union[str, Path],
        equipment_id: Optional[UUID] = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        recorded_at: Optional[datetime] = None,
        metadata: Optional[Dict] = None
    ) -> CSVProcessingStats:
        """
        Загрузить CSV файл в базу данных

        Args:
            file_path: Путь к CSV файлу
            equipment_id: UUID оборудования (если None, определяется автоматически)
            sample_rate: Частота дискретизации в Гц
            recorded_at: Время записи (если None, используется время модификации файла)

        Returns:
            Статистика обработки
        """
        file_path = Path(file_path)
        stats = CSVProcessingStats()

        if not file_path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        self.logger.info(f"Начинаем обработку файла: {file_path}")
        self.logger.info(f"Размер файла: {file_path.stat().st_size / (1024*1024):.1f} MB")

        # Вычисляем хеш файла для дедупликации
        file_hash = calculate_file_hash(file_path)
        self.logger.debug(f"SHA256 файла: {file_hash}")

        # Время записи по умолчанию - время модификации файла
        if recorded_at is None:
            recorded_at = datetime.fromtimestamp(file_path.stat().st_mtime)

        try:
            file_timer_start = datetime.now()
            async with get_async_session() as session:
                # Проверяем, не загружен ли уже этот файл
                from sqlalchemy import select

                existing_signal = await session.execute(
                    select(RawSignal).where(RawSignal.file_hash == file_hash)
                )

                if existing_signal.scalar_one_or_none():
                    self.logger.warning(f"Файл уже загружен (hash: {file_hash[:8]}...)")
                    return stats

                # Определяем оборудование
                if equipment_id is None:
                    equipment = await find_equipment_by_filename(session, file_path.name)
                    if not equipment:
                        raise CSVLoaderError("Не удалось определить оборудование для файла")
                    equipment_id = equipment.id

                # Обрабатываем файл по частям
                async for batch_stats in self._process_csv_file_batches(
                    file_path, equipment_id, sample_rate, recorded_at, file_hash, session, metadata or {}
                ):
                    stats.add_batch_stats(
                        batch_stats['processed_rows'],
                        batch_stats['nan_counts'],
                        batch_stats.get('raw_id'),
                        batch_stats.get('samples_count', 0),
                        batch_stats.get('phases_present', {"R": False, "S": False, "T": False})
                    )

                    self.logger.info(
                        f"Обработана пачка {stats.batches_processed}: "
                        f"{batch_stats['processed_rows']} строк, "
                        f"NaN: R={batch_stats['nan_counts']['R']}, "
                        f"S={batch_stats['nan_counts']['S']}, "
                        f"T={batch_stats['nan_counts']['T']}"
                    )

                await session.commit()

        except Exception as e:
            self.logger.error(f"Ошибка при обработке файла {file_path}: {e}")
            try:
                m.increment_counter('csv_files_processed_total', {'equipment_id': str(equipment_id) if equipment_id else 'unknown', 'status': 'failed'})
            except Exception:
                pass
            raise

        stats.finish()
        # Метрики после успешной обработки
        try:
            m.increment_counter('csv_files_processed_total', {'equipment_id': str(equipment_id), 'status': 'success'})
            m.observe_histogram('csv_processing_duration_seconds', stats.processing_time, {'equipment_id': str(equipment_id)})
            # По фазам количество точек
            for phase in ['R','S','T']:
                if stats.nan_values.get(phase, 0) >= 0:
                    m.increment_counter('data_points_processed_total', {'equipment_id': str(equipment_id), 'phase': phase}, value=float(stats.samples_count_total))
        except Exception:
            pass

        self.logger.info(f"Файл обработан успешно:")
        self.logger.info(f"  - Строк обработано: {stats.processed_rows:,}")
        self.logger.info(f"  - Пачек: {stats.batches_processed}")
        self.logger.info(f"  - Время: {stats.processing_time:.1f} сек")
        self.logger.info(f"  - Скорость: {stats.rows_per_second:,.0f} строк/сек")
        self.logger.info(f"  - NaN значений: R={stats.nan_values['R']:,}, "
                        f"S={stats.nan_values['S']:,}, T={stats.nan_values['T']:,}")

        return stats

    async def _process_csv_file_batches(
        self,
        file_path: Path,
        equipment_id: UUID,
        sample_rate: int,
        recorded_at: datetime,
        file_hash: str,
        session: AsyncSession,
        metadata: Optional[Dict] = None,
    ) -> AsyncGenerator[Dict, None]:
        """
        Обработать CSV файл по пачкам

        Yields:
            Словарь со статистикой обработанной пачки
        """
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)

            # Читаем заголовок (первая строка одна ячейка с тремя именами)
            try:
                header_row = next(reader)
                if not header_row:
                    raise InvalidCSVFormatError("Пустой заголовок")

                # Поддержка как один-столбец-строка, так и стандартного многоколонного CSV
                header_line = header_row[0] if (len(header_row) == 1) else ','.join(header_row)
                _ = parse_csv_header(header_line)

            except StopIteration:
                raise InvalidCSVFormatError("Файл пуст")

            # Обрабатываем данные пачками
            batch_data = {'R': [], 'S': [], 'T': []}
            batch_nan_counts = {'R': 0, 'S': 0, 'T': 0}
            batch_size = 0
            row_number = 1  # 0-я строка — заголовок
            corrupt_logged = 0

            for row in reader:
                row_number += 1

                # Пропускаем полностью пустые строки
                if not row or (len(row) == 1 and not row[0].strip()) or (len(row) == 0):
                    continue

                try:
                    # Парсим строку данных
                    row_line = row[0] if (len(row) == 1) else ','.join(row)
                    values, nan_mask = parse_csv_row(row_line)

                    # Добавляем данные в пачку
                    for i, phase in enumerate(['R', 'S', 'T']):
                        batch_data[phase].append(values[i])
                        if nan_mask[i]:
                            batch_nan_counts[phase] += 1

                    batch_size += 1

                    # Если пачка заполнена, сохраняем
                    batch_t0 = time.time()
                    if batch_size >= self.batch_size:
                        raw_id, samples_count, phases_present = await self._save_batch_to_db(
                            batch_data, equipment_id, sample_rate,
                            recorded_at, file_hash, file_path.name, session, metadata
                        )

                        # Возвращаем статистику пачки
                        yield {
                            'processed_rows': batch_size,
                            'nan_counts': batch_nan_counts.copy(),
                            'raw_id': raw_id,
                            'samples_count': samples_count,
                            'phases_present': phases_present
                        }

                        # Очищаем пачку
                        batch_data = {'R': [], 'S': [], 'T': []}
                        batch_nan_counts = {'R': 0, 'S': 0, 'T': 0}
                        # Метрики по пачке
                        try:
                            m.observe_histogram('csv_batch_rows', float(self.batch_size), {'equipment_id': str(equipment_id)})
                            m.observe_histogram('csv_batch_duration_seconds', float(time.time() - batch_t0), {'equipment_id': str(equipment_id)})
                        except Exception:
                            pass
                        batch_size = 0
                        try:
                            for p in ['R','S','T']:
                                if phases_present.get(p):
                                    m.increment_counter('data_points_processed_total', {'equipment_id': str(equipment_id), 'phase': p}, value=float(samples_count))
                        except Exception:
                            pass

                except Exception as e:
                    if corrupt_logged < 100:
                        self.logger.warning(f"Ошибка в строке {row_number}: {e}")
                        corrupt_logged += 1
                    continue

            # Сохраняем оставшиеся данные
            if batch_size > 0:
                batch_t0 = time.time()
                raw_id, samples_count, phases_present = await self._save_batch_to_db(
                    batch_data, equipment_id, sample_rate,
                    recorded_at, file_hash, file_path.name, session, metadata
                )

                yield {
                    'processed_rows': batch_size,
                    'nan_counts': batch_nan_counts,
                    'raw_id': raw_id,
                    'samples_count': samples_count,
                    'phases_present': phases_present
                }
                # Метрики по финальной пачке
                try:
                    for p in ['R','S','T']:
                        if phases_present.get(p):
                            m.increment_counter('data_points_processed_total', {'equipment_id': str(equipment_id), 'phase': p}, value=float(samples_count))
                    m.observe_histogram('csv_batch_rows', float(batch_size), {'equipment_id': str(equipment_id)})
                    m.observe_histogram('csv_batch_duration_seconds', float(time.time() - batch_t0), {'equipment_id': str(equipment_id)})
                except Exception:
                    pass

    async def _save_batch_to_db(
        self,
        batch_data: Dict[str, List[float]],
        equipment_id: UUID,
        sample_rate: int,
        recorded_at: datetime,
        file_hash: str,
        file_name: str,
        session: AsyncSession,
        metadata: Optional[Dict] = None,
    ) -> Tuple[Optional[UUID], int, Dict[str, bool]]:
        """
        Сохранить пачку данных в базу данных

        Args:
            batch_data: Данные пачки по фазам
            equipment_id: UUID оборудования
            sample_rate: Частота дискретизации
            recorded_at: Время записи
            file_hash: Хеш файла
            file_name: Имя файла
            session: Сессия базы данных
        """
    # Конвертируем в numpy массивы
        phase_arrays = {}
        samples_count = 0

        for phase in ['R', 'S', 'T']:
            data_list = batch_data[phase]
            if data_list:
                arr = np.array(data_list, dtype=np.float32)
                if np.all(np.isnan(arr)):
                    phase_arrays[phase] = None
                else:
                    phase_arrays[phase] = dump_float32_array(arr)
                    samples_count = max(samples_count, arr.size)
            else:
                phase_arrays[phase] = None

        phases_present = {
            'R': phase_arrays['R'] is not None,
            'S': phase_arrays['S'] is not None,
            'T': phase_arrays['T'] is not None,
        }

        if samples_count == 0:
            self.logger.warning("Пачка содержит только NaN значения, пропускаем")
            return None, 0, phases_present

        # Нормализуем recorded_at, поддерживая строку ISO
        rec_at = recorded_at
        if isinstance(rec_at, str):
            try:
                rec_at = datetime.fromisoformat(rec_at)
            except Exception:
                rec_at = datetime.utcnow()

        # Создаем запись в базе данных
        raw_signal = RawSignal(
            equipment_id=equipment_id,
            recorded_at=rec_at,
            sample_rate_hz=sample_rate,
            samples_count=samples_count,
            phase_a=phase_arrays['R'],
            phase_b=phase_arrays['S'],
            phase_c=phase_arrays['T'],
            file_name=file_name,
            file_hash=file_hash,
            meta={
                **(metadata or {}),
                'batch_size': len(batch_data['R']),
                'file_format': 'csv_single_column',
                'phases': ['R', 'S', 'T'],
                'loader_version': '1.1.0'
            }
        )

        session.add(raw_signal)

        # Флашим изменения, но не коммитим (это делается выше)
        await session.flush()
        return raw_signal.id, samples_count, phases_present


# Вспомогательные функции для CLI использования

async def load_csv_files_from_directory(
    directory_path: Union[str, Path],
    pattern: str = "*.csv",
    equipment_id: Optional[UUID] = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE
) -> Dict[str, CSVProcessingStats]:
    """
    Загрузить все CSV файлы из директории

    Args:
        directory_path: Путь к директории
        pattern: Паттерн имен файлов
        equipment_id: UUID оборудования
        sample_rate: Частота дискретизации

    Returns:
        Словарь со статистикой по каждому файлу
    """
    directory_path = Path(directory_path)
    loader = CSVLoader()
    results = {}

    csv_files = list(directory_path.glob(pattern))
    logger.info(f"Найдено {len(csv_files)} CSV файлов в {directory_path}")

    for file_path in csv_files:
        try:
            logger.info(f"Обрабатываем файл: {file_path.name}")
            stats = await loader.load_csv_file(
                file_path=file_path,
                equipment_id=equipment_id,
                sample_rate=sample_rate
            )
            results[file_path.name] = stats

        except Exception as e:
            logger.error(f"Ошибка при обработке {file_path.name}: {e}")
            results[file_path.name] = None

    return results


if __name__ == "__main__":
    # Пример использования
    import argparse

    parser = argparse.ArgumentParser(description="Загрузка CSV файлов с токовыми сигналами")
    parser.add_argument("path", help="Путь к CSV файлу или директории")
    parser.add_argument("--equipment-id", type=str, help="UUID оборудования")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE,
                       help="Частота дискретизации (Гц)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                       help="Размер пачки для загрузки")

    args = parser.parse_args()

    async def main():
        path = Path(args.path)
        equipment_id = UUID(args.equipment_id) if args.equipment_id else None

        if path.is_file():
            # Загружаем один файл
            loader = CSVLoader(batch_size=args.batch_size)
            stats = await loader.load_csv_file(
                file_path=path,
                equipment_id=equipment_id,
                sample_rate=args.sample_rate
            )
            print(f"Файл обработан: {stats.to_dict()}")

        elif path.is_dir():
            # Загружаем все файлы из директории
            results = await load_csv_files_from_directory(
                directory_path=path,
                equipment_id=equipment_id,
                sample_rate=args.sample_rate
            )

            print(f"Обработано файлов: {len(results)}")
            for filename, stats in results.items():
                if stats:
                    print(f"  {filename}: {stats.processed_rows:,} строк")
                else:
                    print(f"  {filename}: ОШИБКА")
        else:
            print(f"Путь не найден: {path}")

    asyncio.run(main())
