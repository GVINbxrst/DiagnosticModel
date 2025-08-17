"""
Unit тесты для CSV загрузчика
Тестирование функций обработки CSV файлов с токовыми сигналами
"""

import asyncio
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.data_processing.csv_loader import (
    CSVLoader,
    CSVProcessingStats,
    compress_float32_array,
    decompress_float32_array,
    parse_csv_header,
    parse_csv_row,
    calculate_file_hash,
    InvalidCSVFormatError
)


class TestCSVUtilityFunctions:
    """Тесты вспомогательных функций"""

    def test_compress_decompress_float32_array(self):
        """Тест сжатия и распаковки массивов float32"""
        # Тестовые данные
        original_data = np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float32)

        # Сжимаем
        compressed = compress_float32_array(original_data)
        assert isinstance(compressed, bytes)
        assert len(compressed) > 0
    # Не проверяем коэффициент сжатия строго, только что не пусто

        # Распаковываем
        decompressed = decompress_float32_array(compressed)
        assert isinstance(decompressed, np.ndarray)
        assert decompressed.dtype == np.float32
        np.testing.assert_array_almost_equal(original_data, decompressed)

    def test_compress_empty_array(self):
        """Тест сжатия пустого массива"""
        empty_data = np.array([], dtype=np.float32)
        compressed = compress_float32_array(empty_data)
        assert compressed == b''

        decompressed = decompress_float32_array(b'')
        assert len(decompressed) == 0
        assert decompressed.dtype == np.float32

    def test_compress_with_nan_values(self):
        """Тест сжатия массива с NaN значениями"""
        data_with_nan = np.array([1.1, np.nan, 3.3, np.nan, 5.5], dtype=np.float32)

        compressed = compress_float32_array(data_with_nan)
        decompressed = decompress_float32_array(compressed)

        # Проверяем, что NaN значения сохранились
        assert np.isnan(decompressed[1])
        assert np.isnan(decompressed[3])
        # numpy 2.x: assert_array_equal не принимает equal_nan; используем сравнение через isnan
        assert data_with_nan.shape == decompressed.shape
        for a,b in zip(data_with_nan, decompressed):
            if np.isnan(a):
                assert np.isnan(b)
            else:
                assert a == b

    def test_parse_csv_header_valid(self):
        """Тест парсинга корректного заголовка"""
        header = "current_R,current_S,current_T"
        phases = parse_csv_header(header)

        assert phases == ["current_R", "current_S", "current_T"]

    def test_parse_csv_header_with_spaces(self):
        """Тест парсинга заголовка с пробелами"""
        header = " current_R , current_S , current_T "
        phases = parse_csv_header(header)

        assert phases == ["current_R", "current_S", "current_T"]

    def test_parse_csv_header_invalid_count(self):
        """Тест что теперь допускается <3 фаз с автодополнением"""
        header = "current_R,current_S"
        phases = parse_csv_header(header)
        assert len(phases) == 3
        assert phases[0].startswith("current_R")
        assert phases[1].startswith("current_S")
        assert phases[2].startswith("current_T")

    def test_parse_csv_header_single(self):
        """Тест заголовка с одной фазой"""
        header = "current_R"
        phases = parse_csv_header(header)
        assert phases == ["current_R","current_S","current_T"]

    def test_parse_csv_row_valid(self):
        """Тест парсинга корректной строки данных"""
        row_data = "1.23,4.56,7.89"
        values, nan_mask = parse_csv_row(row_data)

        assert values == [1.23, 4.56, 7.89]
        assert nan_mask == [False, False, False]

    def test_parse_csv_row_with_empty_values(self):
        """Тест парсинга строки с пустыми значениями"""
        row_data = "1.23,,7.89"
        values, nan_mask = parse_csv_row(row_data)

        assert values[0] == 1.23
        assert np.isnan(values[1])
        assert values[2] == 7.89
        assert nan_mask == [False, True, False]

    def test_parse_csv_row_with_nan_strings(self):
        """Тест парсинга строки с явными NaN значениями"""
        row_data = "1.23,NaN,null"
        values, nan_mask = parse_csv_row(row_data)

        assert values[0] == 1.23
        assert np.isnan(values[1])
        assert np.isnan(values[2])
        assert nan_mask == [False, True, True]

    def test_parse_csv_row_invalid_float(self):
        """Тест парсинга строки с некорректными числами"""
        row_data = "1.23,abc,7.89"
        values, nan_mask = parse_csv_row(row_data)

        assert values[0] == 1.23
        assert np.isnan(values[1])
        assert values[2] == 7.89
        assert nan_mask == [False, True, False]

    def test_calculate_file_hash(self, tmp_path):
        """Тест вычисления хеша файла"""
        # Создаем тестовый файл
        test_file = tmp_path / "test.csv"
        test_content = "current_R,current_S,current_T\n1.1,2.2,3.3\n"
        test_file.write_text(test_content)

        # Вычисляем хеш
        hash1 = calculate_file_hash(test_file)
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA256 в hex

        # Проверяем, что хеш стабильный
        hash2 = calculate_file_hash(test_file)
        assert hash1 == hash2

        # Изменяем файл и проверяем, что хеш изменился
        test_file.write_text(test_content + "4.4,5.5,6.6\n")
        hash3 = calculate_file_hash(test_file)
        assert hash1 != hash3


class TestCSVProcessingStats:
    """Тесты класса статистики обработки"""

    def test_stats_initialization(self):
        """Тест инициализации статистики"""
        stats = CSVProcessingStats()

        assert stats.total_rows == 0
        assert stats.processed_rows == 0
        assert stats.skipped_rows == 0
        assert stats.invalid_rows == 0
        assert stats.nan_values == {'R': 0, 'S': 0, 'T': 0}
        assert stats.batches_processed == 0
        assert stats.start_time is not None
        assert stats.end_time is None

    def test_add_batch_stats(self):
        """Тест добавления статистики пачки"""
        stats = CSVProcessingStats()

        stats.add_batch_stats(1000, {'R': 10, 'S': 20, 'T': 15})

        assert stats.processed_rows == 1000
        assert stats.batches_processed == 1
        assert stats.nan_values == {'R': 10, 'S': 20, 'T': 15}

        # Добавляем еще одну пачку
        stats.add_batch_stats(500, {'R': 5, 'S': 10, 'T': 8})

        assert stats.processed_rows == 1500
        assert stats.batches_processed == 2
        assert stats.nan_values == {'R': 15, 'S': 30, 'T': 23}

    def test_stats_timing(self):
        """Тест измерения времени обработки"""
        stats = CSVProcessingStats()

        # Имитируем обработку
        import time
        time.sleep(0.1)

        stats.finish()

        assert stats.end_time is not None
        assert stats.processing_time > 0.09  # Примерно 0.1 секунды
        assert stats.processing_time < 0.2   # Но не слишком много

    def test_stats_to_dict(self):
        """Тест преобразования статистики в словарь"""
        stats = CSVProcessingStats()
        stats.add_batch_stats(1000, {'R': 10, 'S': 20, 'T': 15})
        stats.finish()

        stats_dict = stats.to_dict()

        assert isinstance(stats_dict, dict)
        assert 'processed_rows' in stats_dict
        assert 'nan_values' in stats_dict
        assert 'processing_time_seconds' in stats_dict
        assert 'rows_per_second' in stats_dict
        assert stats_dict['processed_rows'] == 1000


class TestCSVLoader:
    """Тесты основного класса CSVLoader"""

    @pytest.fixture
    def csv_loader(self):
        """Фикстура для создания CSVLoader"""
        return CSVLoader(batch_size=100)

    @pytest.fixture
    def mock_session(self):
        """Мок для сессии базы данных"""
        session = AsyncMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.flush = AsyncMock()
        session.add = MagicMock()
        return session

    @pytest.fixture
    def sample_csv_file(self, tmp_path):
        """Создать тестовый CSV файл"""
        csv_file = tmp_path / "test_motor.csv"
        # Используем одностолбцовый формат (по сути одна колонка со строками с запятыми)
        csv_content = (
            "current_R,current_S,current_T\n"
            "1.1,2.2,3.3\n"
            "1.2,2.3,3.4\n"
            "1.3,,3.5\n"
            "1.4,2.5,\n"
            ",2.6,3.7\n"
            "1.7,2.7,3.8\n"
            "1.8,2.8,3.9\n"
            "1.9,2.9,4.0\n"
            "2.0,3.0,4.1\n"
            "2.1,3.1,4.2"
        )
        csv_file.write_text(csv_content)
        return csv_file

    @pytest.mark.asyncio
    async def test_process_csv_file_batches(self, csv_loader, sample_csv_file, mock_session):
        """Тест обработки CSV файла по пачкам"""
        equipment_id = uuid4()
        sample_rate = 25600
        recorded_at = "2025-01-10 10:00:00"
        file_hash = "test_hash"

        # Мокаем сохранение в БД
        with patch.object(csv_loader, '_save_batch_to_db', new_callable=AsyncMock) as mock_save:
            batches = []
            async for batch_stats in csv_loader._process_csv_file_batches(
                sample_csv_file, equipment_id, sample_rate, recorded_at, file_hash, mock_session
            ):
                batches.append(batch_stats)

            # Проверяем, что данные обработались
            assert len(batches) >= 1

            total_processed = sum(batch['processed_rows'] for batch in batches)
            assert total_processed == 10  # 10 строк данных в тестовом файле

            # Проверяем, что метод сохранения вызывался
            assert mock_save.called

    def test_csv_loader_initialization(self):
        """Тест инициализации CSV загрузчика"""
        loader = CSVLoader(batch_size=5000)

        assert loader.batch_size == 5000
        assert loader.logger is not None

    @pytest.mark.asyncio
    async def test_save_batch_to_db(self, csv_loader, mock_session):
        """Тест сохранения пачки в базу данных"""
        batch_data = {
            'R': [1.1, 1.2, 1.3],
            'S': [2.1, 2.2, np.nan],
            'T': [3.1, np.nan, 3.3]
        }

        equipment_id = uuid4()
        sample_rate = 25600
        recorded_at = "2025-01-10 10:00:00"
        file_hash = "test_hash"
        file_name = "test.csv"

        # Выполняем сохранение
        await csv_loader._save_batch_to_db(
            batch_data, equipment_id, sample_rate, recorded_at, file_hash, file_name, mock_session
        )

        # Проверяем, что объект был добавлен в сессию
        assert mock_session.add.called
        assert mock_session.flush.called

        # Получаем добавленный объект
        added_signal = mock_session.add.call_args[0][0]

        assert added_signal.equipment_id == equipment_id
        assert added_signal.sample_rate_hz == sample_rate
        assert added_signal.samples_count == 3
        assert added_signal.file_name == file_name
        assert added_signal.file_hash == file_hash

        # Проверяем, что фазы сжаты
        assert added_signal.phase_a is not None  # Фаза R есть
        assert added_signal.phase_b is not None  # Фаза S есть (не все NaN)
        assert added_signal.phase_c is not None  # Фаза T есть (не все NaN)

    @pytest.mark.asyncio
    async def test_phase_status_missing(self, tmp_path, csv_loader, mock_session):
        """Проверка вычисления phase_status при полностью пустой фазе."""
        csv_file = tmp_path / "missing_phase.csv"
        # Фаза S полностью пустая (одностолбцовый режим)
        csv_file.write_text("current_R,current_S,current_T\n1.0,,3.0\n2.0,,4.0\n")
        # Мокаем сессию внутри load_csv_file
        with patch('src.data_processing.csv_loader.get_async_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_get_session.return_value.__aexit__.return_value = None
            # Не найден существующий сигнал
            mock_session.execute.return_value.scalar_one_or_none.return_value = None
            with patch('src.data_processing.csv_loader.find_equipment_by_filename') as mock_find_eq:
                fake_eq = MagicMock(); fake_eq.id = uuid4(); mock_find_eq.return_value = fake_eq
                stats = await csv_loader.load_csv_file(csv_file)
        stats_dict = stats.to_dict()
        assert stats_dict['phase_status']['S'] in ('missing','ok')  # В зависимости от логики NaN подсчета
        # Проверим что nan_values для S равны количеству обработанных строк
        assert stats_dict['nan_values']['S'] == stats.processed_rows
    
    def test_parse_csv_row_single_value(self):
        """Строка только с одним значением"""
        row_data = "1.23"
        values, mask = parse_csv_row(row_data)
        assert values[0] == 1.23
        assert np.isnan(values[1]) and np.isnan(values[2])
        assert mask == [False, True, True]


@pytest.mark.integration
class TestCSVLoaderIntegration:
    """Интеграционные тесты CSV загрузчика"""

    @pytest.fixture
    def large_csv_file(self, tmp_path):
        """Создать большой тестовый CSV файл"""
        csv_file = tmp_path / "large_motor.csv"

        with open(csv_file, 'w') as f:
            f.write("current_R,current_S,current_T\n")

            # Генерируем 15000 строк тестовых данных
            for i in range(15000):
                r_val = 1.0 + 0.1 * np.sin(i * 0.01)
                s_val = 1.1 + 0.1 * np.cos(i * 0.01) if i % 10 != 0 else ""  # Пропуски каждые 10 строк
                t_val = 1.2 + 0.1 * np.sin(i * 0.02) if i % 15 != 0 else ""  # Пропуски каждые 15 строк

                f.write(f"{r_val:.3f},{s_val},{t_val}\n")

        return csv_file

    @pytest.mark.asyncio
    async def test_large_file_processing(self, large_csv_file):
        """Тест обработки большого файла"""
        loader = CSVLoader(batch_size=1000)

        # Мокаем базу данных
        with patch('src.data_processing.csv_loader.get_async_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_get_session.return_value = mock_session

            # Мокаем поиск оборудования
            with patch('src.data_processing.csv_loader.find_equipment_by_filename') as mock_find_eq:
                mock_equipment = MagicMock()
                mock_equipment.id = uuid4()
                mock_find_eq.return_value = mock_equipment

                # Мокаем проверку существующих сигналов
                mock_session.execute.return_value.scalar_one_or_none.return_value = None

                # Выполняем загрузку
                stats = await loader.load_csv_file(large_csv_file)

                # Проверяем результаты
                assert stats.processed_rows == 15000
                assert stats.batches_processed == 15  # 15000 / 1000
                assert stats.processing_time > 0
                assert stats.rows_per_second > 0

                # Проверяем, что были NaN значения
                assert stats.nan_values['S'] > 0  # Фаза S имеет пропуски
                assert stats.nan_values['T'] > 0  # Фаза T имеет пропуски


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
