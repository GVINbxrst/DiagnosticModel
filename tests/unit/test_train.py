"""
Unit тесты для модуля обучения моделей аномалий
Тестирование ML компонентов для детекции аномалий в токовых сигналах
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.ml.train import (
    FeaturePreprocessor,
    AnomalyDetectionModels,
    AnomalyModelTrainer,
    InsufficientDataError,
    DEFAULT_CONTAMINATION,
    MIN_SAMPLES_FOR_TRAINING
)


class TestFeaturePreprocessor:
    """Тесты предобработчика признаков"""

    @pytest.fixture
    def preprocessor(self):
        return FeaturePreprocessor()

    @pytest.fixture
    def sample_features_df(self):
        """Создать тестовый DataFrame с признаками"""
        np.random.seed(42)
        n_samples = 200

        data = {
            'rms_a': np.random.normal(5.0, 1.0, n_samples),
            'rms_b': np.random.normal(5.2, 1.1, n_samples),
            'rms_c': np.random.normal(4.8, 0.9, n_samples),
            'crest_a': np.random.normal(1.5, 0.2, n_samples),
            'crest_b': np.random.normal(1.6, 0.25, n_samples),
            'crest_c': np.random.normal(1.4, 0.15, n_samples),
            'kurt_a': np.random.normal(3.0, 0.5, n_samples),
            'kurt_b': np.random.normal(3.1, 0.6, n_samples),
            'kurt_c': np.random.normal(2.9, 0.4, n_samples),
            'skew_a': np.random.normal(0.0, 0.3, n_samples),
            'skew_b': np.random.normal(0.1, 0.35, n_samples),
            'skew_c': np.random.normal(-0.05, 0.25, n_samples)
        }

        # Добавляем несколько NaN значений
        for col in ['rms_b', 'kurt_c']:
            mask = np.random.choice(n_samples, size=5, replace=False)
            for i in mask:
                data[col][i] = np.nan

        return pd.DataFrame(data)

    def test_prepare_features_basic(self, preprocessor, sample_features_df):
        """Тест базовой подготовки признаков"""
        X_scaled, feature_names = preprocessor.prepare_features_for_training(sample_features_df)

        # Проверяем размерность результата
        assert X_scaled.shape[0] == len(sample_features_df)
        assert X_scaled.shape[1] > 0
        assert len(feature_names) == X_scaled.shape[1]

        # Проверяем, что данные нормализованы (примерно)
        assert np.abs(np.mean(X_scaled)) < 0.5  # Среднее близко к 0
        assert 0.5 < np.std(X_scaled) < 2.0     # Стандартное отклонение разумное

        # Проверяем, что нет NaN
        assert not np.any(np.isnan(X_scaled))

    def test_handle_missing_values(self, preprocessor):
        """Тест обработки пропущенных значений"""
        # DataFrame с пропущенными значениями
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, np.nan, 4.0, 5.0],
            'feature2': [np.nan, 2.0, 3.0, np.nan, 5.0],
            'feature3': [1.0, 2.0, 3.0, 4.0, 5.0]
        })

        result = preprocessor._handle_missing_values(df)

        # Проверяем, что NaN заполнены
        assert not result.isna().any().any()

        # Проверяем, что значения разумные
        assert all(result['feature3'] == df['feature3'])  # Колонка без NaN не изменилась

    def test_scale_features(self, preprocessor):
        """Тест нормализации признаков"""
        # Создаем тестовые данные с разными масштабами
        df = pd.DataFrame({
            'small_feature': np.random.normal(0.1, 0.01, 100),
            'large_feature': np.random.normal(1000, 100, 100),
            'normal_feature': np.random.normal(5, 1, 100)
        })

        scaled = preprocessor._scale_features(df)

        # Проверяем, что скейлер обучен
        assert preprocessor.feature_scaler is not None

        # Проверяем размерность
        assert scaled.shape == df.shape

        # Проверяем, что данные приблизительно нормализованы
        # RobustScaler использует медиану и IQR, поэтому результат может отличаться от StandardScaler
        assert np.all(np.abs(np.median(scaled, axis=0)) < 0.5)

    def test_feature_selection_by_variance(self, preprocessor):
        """Тест отбора признаков по дисперсии"""
        # Создаем признаки с разной дисперсией
        df = pd.DataFrame({
            'high_var': np.random.normal(0, 10, 100),     # Высокая дисперсия
            'medium_var': np.random.normal(0, 1, 100),    # Средняя дисперсия
            'low_var': np.random.normal(0, 0.01, 100),    # Низкая дисперсия
            'zero_var': np.ones(100)                      # Нулевая дисперсия
        })

        result_df, selected_features = preprocessor._select_features_by_variance(df, threshold=0.1)

        # Проверяем, что признаки с низкой дисперсией исключены
        assert 'high_var' in selected_features
        assert 'medium_var' in selected_features
        assert 'low_var' not in selected_features
        assert 'zero_var' not in selected_features

        assert result_df.shape[1] == len(selected_features)

    def test_transform_features(self, preprocessor, sample_features_df):
        """Тест применения обученного препроцессора к новым данным"""
        # Обучаем препроцессор
        X_train, feature_names = preprocessor.prepare_features_for_training(sample_features_df)

        # Создаем новые данные
        new_data = sample_features_df.iloc[:10].copy()

        # Применяем трансформацию
        X_new = preprocessor.transform_features(new_data)

        # Проверяем размерность
        assert X_new.shape[0] == 10
        assert X_new.shape[1] == X_train.shape[1]

        # Проверяем, что нет NaN
        assert not np.any(np.isnan(X_new))

    def test_insufficient_features_error(self, preprocessor):
        """Тест ошибки при недостаточном количестве признаков"""
        # DataFrame без признаков из нужного списка
        df = pd.DataFrame({
            'unknown_feature1': [1, 2, 3],
            'unknown_feature2': [4, 5, 6]
        })

        with pytest.raises(ValueError, match="Не найдено признаков для обучения"):
            preprocessor.prepare_features_for_training(df)


@pytest.mark.skip(reason="IsolationForest удалён; тест деактивирован")
async def test_mvp_train_isolation_forest(monkeypatch, tmp_path):  # pragma: no cover
    pass


class TestAnomalyDetectionModels:
    """Тесты моделей обнаружения аномалий"""

    @pytest.fixture
    def models(self):
        return AnomalyDetectionModels(random_state=42)

    @pytest.fixture
    def sample_data(self):
        """Создать тестовые данные для обучения"""
        np.random.seed(42)

        # Нормальные данные
        normal_data = np.random.multivariate_normal(
            mean=[0, 0, 0],
            cov=[[1, 0.3, 0.1], [0.3, 1, 0.2], [0.1, 0.2, 1]],
            size=200
        )

        # Аномальные данные
        anomaly_data = np.random.multivariate_normal(
            mean=[5, 5, 5],
            cov=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            size=20
        )

        # Объединяем
        X = np.vstack([normal_data, anomaly_data])
        y_true = np.hstack([np.ones(200), -np.ones(20)])  # 1 = нормальные, -1 = аномалии

        return X, y_true

    @pytest.mark.skip(reason="IsolationForest удалён; тест деактивирован")
    def test_train_isolation_forest(self, models, sample_data):  # pragma: no cover
        pass

    def test_train_dbscan(self, models, sample_data):
        """Тест обучения DBSCAN"""
        X, y_true = sample_data

        labels = models.train_dbscan(X, eps=1.0, min_samples=5)

        # Проверяем, что модель обучена
        assert models.dbscan is not None
        assert models.dbscan_labels is not None

        # Проверяем размерность меток
        assert len(labels) == len(X)

        # Проверяем, что найдены кластеры
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        assert n_clusters >= 1

    def test_train_pca(self, models, sample_data):
        """Тест обучения PCA"""
        X, y_true = sample_data

        components = models.train_pca(X, n_components=2)

        # Проверяем, что модель обучена
        assert models.pca is not None
        assert models.pca_components is not None

        # Проверяем размерность компонент
        assert components.shape == (len(X), 2)

        # Проверяем, что объяснена разумная доля дисперсии
        total_variance = np.sum(models.pca.explained_variance_ratio_)
        assert total_variance > 0.5  # Минимум 50% дисперсии

    @pytest.mark.skip(reason="IsolationForest удалён; тест деактивирован")
    def test_feature_importance_isolation_forest(self, models, sample_data):  # pragma: no cover
        pass

    def test_pca_feature_contribution(self, models, sample_data):
        """Тест вычисления вклада признаков в PCA компоненты"""
        X, y_true = sample_data

        # Обучаем PCA
        models.train_pca(X, n_components=2)

        # Вычисляем вклад признаков
        feature_names = ['feature_1', 'feature_2', 'feature_3']
        contributions = models.get_pca_feature_contribution(feature_names)

        # Проверяем результат
        assert 'PC1' in contributions
        assert 'PC2' in contributions

        for pc_name, pc_contrib in contributions.items():
            assert len(pc_contrib) == len(feature_names)
            assert all(isinstance(v, (int, float)) for v in pc_contrib.values())
            assert all(v >= 0 for v in pc_contrib.values())


class TestAnomalyModelTrainer:
    """Тесты основного класса обучения моделей"""

    @pytest.fixture
    def trainer(self, tmp_path):
        return AnomalyModelTrainer(models_path=tmp_path / "test_models")

    @pytest.fixture
    def mock_features_data(self):
        """Мок данных признаков из базы данных"""
        n_samples = 150
        np.random.seed(42)

        data = []
        for i in range(n_samples):
            feature_dict = {
                'id': uuid4(),
                'equipment_id': uuid4(),
                'window_start': datetime.now() - timedelta(hours=i),
                'window_end': datetime.now() - timedelta(hours=i) + timedelta(minutes=1),

                # Статистические признаки фазы A
                'rms_a': np.random.normal(5.0, 1.0),
                'crest_a': np.random.normal(1.5, 0.2),
                'kurt_a': np.random.normal(3.0, 0.5),
                'skew_a': np.random.normal(0.0, 0.3),
                'mean_a': np.random.normal(4.8, 0.8),
                'std_a': np.random.normal(1.2, 0.2),
                'min_a': np.random.normal(2.0, 0.5),
                'max_a': np.random.normal(8.0, 1.5),

                # Статистические признаки фазы B
                'rms_b': np.random.normal(5.2, 1.1),
                'crest_b': np.random.normal(1.6, 0.25),
                'kurt_b': np.random.normal(3.1, 0.6),
                'skew_b': np.random.normal(0.1, 0.35),
                'mean_b': np.random.normal(5.0, 0.9),
                'std_b': np.random.normal(1.3, 0.25),
                'min_b': np.random.normal(2.2, 0.6),
                'max_b': np.random.normal(8.5, 1.8),

                # Статистические признаки фазы C (с некоторыми None)
                'rms_c': np.random.normal(4.8, 0.9) if i % 10 != 0 else None,
                'crest_c': np.random.normal(1.4, 0.15) if i % 10 != 0 else None,
                'kurt_c': np.random.normal(2.9, 0.4) if i % 10 != 0 else None,
                'skew_c': np.random.normal(-0.05, 0.25) if i % 10 != 0 else None,
                'mean_c': np.random.normal(4.5, 0.7) if i % 10 != 0 else None,
                'std_c': np.random.normal(1.1, 0.2) if i % 10 != 0 else None,
                'min_c': np.random.normal(1.8, 0.4) if i % 10 != 0 else None,
                'max_c': np.random.normal(7.5, 1.2) if i % 10 != 0 else None,
            }
            data.append(feature_dict)

        return pd.DataFrame(data)

    @pytest.mark.asyncio
    async def test_load_training_data_mock(self, trainer):
        """Тест загрузки данных для обучения (мок)"""
        # Мокаем базу данных
        with patch('src.ml.train.get_async_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_get_session.return_value = mock_session

            # Создаем мок данных
            mock_features = []
            for i in range(200):
                mock_feature = MagicMock()
                mock_feature.id = uuid4()
                mock_feature.window_start = datetime.now() - timedelta(hours=i)
                mock_feature.window_end = datetime.now() - timedelta(hours=i) + timedelta(minutes=1)
                mock_feature.rms_a = 5.0 + 0.1 * i
                mock_feature.crest_a = 1.5
                mock_feature.kurt_a = 3.0
                mock_feature.skew_a = 0.0

                # Мок для raw_signal.equipment_id
                mock_feature.raw_signal.equipment_id = uuid4()

                # Добавляем все остальные атрибуты
                for attr in ['rms_b', 'rms_c', 'crest_b', 'crest_c', 'kurt_b', 'kurt_c',
                           'skew_b', 'skew_c', 'mean_a', 'mean_b', 'mean_c',
                           'std_a', 'std_b', 'std_c', 'min_a', 'min_b', 'min_c',
                           'max_a', 'max_b', 'max_c']:
                    setattr(mock_feature, attr, 1.0 + 0.01 * i)

                mock_features.append(mock_feature)

            # Настраиваем мок сессии
            mock_result = MagicMock()
            mock_result.scalars.return_value.all.return_value = mock_features
            mock_session.execute.return_value = mock_result

            # Выполняем тест
            df = await trainer.load_training_data()

            # Проверяем результат
            assert len(df) == 200
            assert 'rms_a' in df.columns
            assert 'crest_a' in df.columns
            assert not df['rms_a'].isna().all()

    def test_analyze_feature_importance(self, trainer, mock_features_data):
        """Тест анализа важности признаков (PCA-only после удаления IsolationForest)"""
        trainer.training_data = mock_features_data
        X, feature_names = trainer.preprocessor.prepare_features_for_training(mock_features_data)
        trainer.feature_names = feature_names
        trainer.models.train_pca(X)
        importance_analysis = trainer.analyze_feature_importance(top_n=5)
        assert 'pca_contributions' in importance_analysis
        pca_analysis = importance_analysis['pca_contributions']
        assert 'PC1' in pca_analysis
        assert 'PC2' in pca_analysis

    def test_save_models(self, trainer, mock_features_data):
        """Тест сохранения моделей"""
        # Подготавливаем данные и обучаем модели
        trainer.training_data = mock_features_data
        X, feature_names = trainer.preprocessor.prepare_features_for_training(mock_features_data)
        trainer.feature_names = feature_names

        trainer.models.train_dbscan(X)
        trainer.models.train_pca(X)

        # Сохраняем модели
        version = trainer.save_models()

        # Проверяем, что файлы созданы
        version_dir = trainer.models_path / version
        assert version_dir.exists()

        # Проверяем наличие файлов моделей
        assert (version_dir / "dbscan.pkl").exists()
        assert (version_dir / "pca.pkl").exists()
        assert (version_dir / "preprocessor.pkl").exists()

        # Проверяем метаданные
        metadata_path = version_dir / "metadata.json"
        assert metadata_path.exists()

        # Проверяем latest симлинк/копию
        latest_path = trainer.models_path / "latest"
        assert latest_path.exists()

    def test_insufficient_data_error(self, trainer):
        """Тест ошибки при недостаточном количестве данных"""
        # Создаем мало данных
        small_data = pd.DataFrame({
            'rms_a': [1, 2, 3],
            'crest_a': [1.1, 1.2, 1.3]
        })

        with patch.object(trainer, 'load_training_data', return_value=small_data):
            with pytest.raises(InsufficientDataError):
                # Пытаемся подготовить данные
                trainer.preprocessor.prepare_features_for_training(small_data)


@pytest.mark.integration
class TestAnomalyModelTrainerIntegration:
    """Интеграционные тесты обучения моделей"""

    @pytest.fixture
    def realistic_motor_data(self):
        """Создать реалистичные данные двигателя для тестирования"""
        np.random.seed(42)
        n_samples = 500

        # Нормальный режим работы (80% данных)
        n_normal = int(0.8 * n_samples)
        normal_data = {
            'rms_a': np.random.normal(5.0, 0.5, n_normal),
            'rms_b': np.random.normal(5.2, 0.6, n_normal),
            'rms_c': np.random.normal(4.8, 0.4, n_normal),
            'crest_a': np.random.normal(1.4, 0.1, n_normal),
            'crest_b': np.random.normal(1.5, 0.12, n_normal),
            'crest_c': np.random.normal(1.3, 0.08, n_normal),
        }

        # Аномальный режим (20% данных) - повышенные значения
        n_anomaly = n_samples - n_normal
        anomaly_data = {
            'rms_a': np.random.normal(8.0, 1.0, n_anomaly),      # Повышенный RMS
            'rms_b': np.random.normal(8.5, 1.2, n_anomaly),
            'rms_c': np.random.normal(7.8, 0.8, n_anomaly),
            'crest_a': np.random.normal(2.5, 0.3, n_anomaly),    # Повышенный Crest Factor
            'crest_b': np.random.normal(2.8, 0.4, n_anomaly),
            'crest_c': np.random.normal(2.2, 0.2, n_anomaly),
        }

        # Объединяем данные
        data = {}
        for key in normal_data.keys():
            data[key] = np.concatenate([normal_data[key], anomaly_data[key]])

        # Добавляем дополнительные признаки
        for phase in ['a', 'b', 'c']:
            data[f'kurt_{phase}'] = np.random.normal(3.0, 0.5, n_samples)
            data[f'skew_{phase}'] = np.random.normal(0.0, 0.3, n_samples)
            data[f'mean_{phase}'] = data[f'rms_{phase}'] * 0.9  # Коррелированные данные
            data[f'std_{phase}'] = data[f'rms_{phase}'] * 0.2
            data[f'min_{phase}'] = data[f'rms_{phase}'] - 2 * data[f'std_{phase}']
            data[f'max_{phase}'] = data[f'rms_{phase}'] + 2 * data[f'std_{phase}']

        # Создаем метки для проверки качества (истинные аномалии)
        y_true = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)])

        df = pd.DataFrame(data)
        return df, y_true

    def test_realistic_anomaly_detection(self, realistic_motor_data, tmp_path):
        """Тест детекции аномалий на реалистичных данных двигателя"""
        df, y_true = realistic_motor_data

        # Создаем тренер
        trainer = AnomalyModelTrainer(models_path=tmp_path / "test_models")
        trainer.training_data = df

        # Подготавливаем данные
        X, feature_names = trainer.preprocessor.prepare_features_for_training(df)
        trainer.feature_names = feature_names

        # Обучаем модели
        dbscan_labels = trainer.models.train_dbscan(X, eps=1.0, min_samples=10)
        pca_components = trainer.models.train_pca(X)
    # IsolationForest проверка пропущена (legacy)

        # DBSCAN должен найти кластеры
        n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        assert n_clusters >= 1

        # PCA должен объяснить разумную долю дисперсии
        explained_variance = np.sum(trainer.models.pca.explained_variance_ratio_)
        assert explained_variance > 0.6  # Минимум 60%

    # Анализ важности IsolationForest удалён – пропуск


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
