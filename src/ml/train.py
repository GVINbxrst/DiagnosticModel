"""
Модуль обучения моделей аномалий для диагностики двигателей

Этот модуль реализует несколько подходов к обнаружению аномалий:
- Isolation Forest для выявления выбросов
- DBSCAN для кластеризации и обнаружения аномальных паттернов
- PCA и t-SNE для визуализации данных
- Анализ важности признаков для интерпретации результатов
"""

import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.settings import get_settings
from src.database.connection import get_async_session
from src.database.models import Feature, Equipment, RawSignal
from src.utils.logger import get_logger

# Настройки
settings = get_settings()
logger = get_logger(__name__)

# Константы
DEFAULT_CONTAMINATION = 0.1  # Ожидаемая доля аномалий
DEFAULT_RANDOM_STATE = 42
MIN_SAMPLES_FOR_TRAINING = 100

def load_latest_models() -> Dict[str, object]:
    """Загрузка последней IsolationForest модели (минимально для worker.tasks).

    Ожидает manifest.json в models/anomaly. Возвращает словарь с ключами
    'isolation_forest' и 'preprocessor' (scaler) если найдено, иначе пустой dict.
    """
    model_dir = settings.models_path / 'anomaly'
    manifest_path = model_dir / 'manifest.json'
    if not manifest_path.exists():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        # Ищем самый поздний updated_at
        latest_key = None
        latest_ts = None
        for k, v in manifest.items():
            ts = v.get('updated_at')
            if ts:
                try:
                    dt = datetime.fromisoformat(ts)
                except Exception:
                    dt = datetime.min
            else:
                dt = datetime.min
            if latest_ts is None or dt > latest_ts:
                latest_ts = dt
                latest_key = k
        if not latest_key:
            return {}
        model_path = Path(manifest[latest_key]['path'])
        if not model_path.exists():
            return {}
        data = joblib.load(model_path)
        model = data.get('model')
        scaler = data.get('scaler')
        return {'isolation_forest': model, 'preprocessor': scaler}
    except Exception as e:
        logger.warning(f"Не удалось загрузить модели: {e}")
        return {}


class AnomalyModelError(Exception):
    """Базовое исключение для моделей аномалий"""
    pass


class InsufficientDataError(AnomalyModelError):
    """Исключение для недостаточного количества данных"""
    pass


# === Minimal MVP function required by contract ===
async def train_isolation_forest(output_path: Optional[str] = None, n_estimators: int = 100) -> Path:
    """MVP обучение IsolationForest и сохранение модели.

    Шаги (упрощённо):
      1. Загружаем признаки (таблица Feature) – только статистика RMS/crest/kurt/skew/mean/std/min/max.
      2. Заполняем NaN медианами; если колонка полностью NaN – отбрасываем.
      3. Масштабируем StandardScaler.
      4. Обучаем IsolationForest(random_state=42, n_estimators=n_estimators).
      5. Сохраняем модель и scaler в models/anomaly/isolation_forest_v1.pkl (joblib).
      6. Обновляем/создаём models/manifest.json (append/update entry).

    Возвращает путь к pkl файлу. Без сложной стратификации и визуализаций.
    """
    model_dir = Path(output_path) if output_path else (settings.models_path / 'anomaly')
    model_dir.mkdir(parents=True, exist_ok=True)

    stat_cols = [
        'rms_a','rms_b','rms_c',
        'crest_a','crest_b','crest_c',
        'kurt_a','kurt_b','kurt_c',
        'skew_a','skew_b','skew_c',
        'mean_a','mean_b','mean_c',
        'std_a','std_b','std_c',
        'min_a','min_b','min_c',
        'max_a','max_b','max_c'
    ]

    async with get_async_session() as session:
        from sqlalchemy import select
        result = await session.execute(select(Feature))
        rows = result.scalars().all()

    if not rows:
        raise InsufficientDataError("Нет данных Feature для обучения")

    # Формируем DataFrame
    import pandas as pd
    records = []
    for r in rows:
        rec = {c: getattr(r, c, None) for c in stat_cols}
        records.append(rec)
    df = pd.DataFrame(records)

    # Отбрасываем пустые колонки
    non_empty = [c for c in stat_cols if c in df.columns and not df[c].isna().all()]
    if not non_empty:
        raise InsufficientDataError("Все статистические признаки пусты")
    df = df[non_empty]

    # Заполняем NaN медианами
    df = df.fillna(df.median()).fillna(0)

    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest

    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)

    model = IsolationForest(n_estimators=n_estimators, random_state=42, contamination='auto')
    model.fit(X)

    # Сохраняем
    import joblib, json
    model_path = model_dir / 'isolation_forest_v1.pkl'
    joblib.dump({'model': model, 'scaler': scaler, 'features': non_empty}, model_path, compress=3)

    manifest_path = model_dir / 'manifest.json'
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        except Exception:
            manifest = {}
    manifest['isolation_forest_v1'] = {
        'path': str(model_path),
        'n_features': len(non_empty),
        'n_estimators': n_estimators,
        'updated_at': datetime.utcnow().isoformat()
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')

    logger.info(f"IsolationForest модель сохранена: {model_path}")
    return model_path


class FeaturePreprocessor:
    """Предобработчик признаков для ML моделей"""

    def __init__(self):
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.feature_scaler = None
        self.feature_names = None
        self.selected_features = None

    def prepare_features_for_training(
        self,
        features_df: pd.DataFrame,
        feature_selection_method: str = 'variance',
        variance_threshold: float = 0.01
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Подготовить признаки для обучения ML моделей

        Args:
            features_df: DataFrame с признаками
            feature_selection_method: Метод отбора признаков
            variance_threshold: Порог дисперсии для отбора признаков

        Returns:
            Кортеж (обработанные признаки, список названий признаков)
        """
        # Извлекаем статистические признаки
        feature_columns = [
            'rms_a', 'rms_b', 'rms_c',
            'crest_a', 'crest_b', 'crest_c',
            'kurt_a', 'kurt_b', 'kurt_c',
            'skew_a', 'skew_b', 'skew_c',
            'mean_a', 'mean_b', 'mean_c',
            'std_a', 'std_b', 'std_c',
            'min_a', 'min_b', 'min_c',
            'max_a', 'max_b', 'max_c'
        ]

        # Отбираем только существующие колонки
        available_features = [col for col in feature_columns if col in features_df.columns]

        if len(available_features) == 0:
            raise ValueError("Не найдено признаков для обучения")

        # Извлекаем данные признаков
        X = features_df[available_features].copy()

        # Обрабатываем пропущенные значения
        X = self._handle_missing_values(X)

        # Отбираем признаки по дисперсии
        if feature_selection_method == 'variance':
            X, selected_features = self._select_features_by_variance(X, variance_threshold)
        else:
            selected_features = list(X.columns)

        # Нормализуем признаки
        X_scaled = self._scale_features(X)

        self.feature_names = available_features
        self.selected_features = selected_features

        self.logger.info(
            f"Подготовлено {X_scaled.shape[0]} образцов с {X_scaled.shape[1]} признаками"
        )

        return X_scaled, selected_features

    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Обработать пропущенные значения"""
        # Заполняем NaN медианными значениями
        X_filled = X.fillna(X.median())

        # Если все еще есть NaN (все значения были NaN), заполняем нулями
        X_filled = X_filled.fillna(0)

        missing_ratio = X.isna().sum().sum() / (X.shape[0] * X.shape[1])
        if missing_ratio > 0:
            self.logger.warning(f"Заполнено {missing_ratio:.2%} пропущенных значений")

        return X_filled

    def _select_features_by_variance(
        self,
        X: pd.DataFrame,
        threshold: float
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Отбор признаков по дисперсии"""
        # Вычисляем дисперсию каждого признака
        variances = X.var()

        # Отбираем признаки с дисперсией выше порога
        high_variance_features = variances[variances > threshold].index.tolist()

        if len(high_variance_features) == 0:
            # Если все признаки имеют низкую дисперсию, берем топ-10
            high_variance_features = variances.nlargest(min(10, len(variances))).index.tolist()
            self.logger.warning(f"Все признаки имеют низкую дисперсию, выбрано {len(high_variance_features)}")

        selected_X = X[high_variance_features]

        self.logger.info(
            f"Отобрано {len(high_variance_features)} признаков из {len(X.columns)} "
            f"по критерию дисперсии > {threshold}"
        )

        return selected_X, high_variance_features

    def _scale_features(self, X: pd.DataFrame) -> np.ndarray:
        """Нормализация признаков"""
        # Используем RobustScaler для устойчивости к выбросам
        self.feature_scaler = RobustScaler()
        X_scaled = self.feature_scaler.fit_transform(X)

        return X_scaled

    def transform_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """Применить обученную предобработку к новым данным"""
        if self.feature_scaler is None or self.selected_features is None:
            raise ValueError("Препроцессор не обучен")

        # Отбираем те же признаки
        X = features_df[self.selected_features].copy()

        # Обрабатываем пропущенные значения
        X = self._handle_missing_values(X)

        # Применяем нормализацию
        X_scaled = self.feature_scaler.transform(X)

        return X_scaled


class AnomalyDetectionModels:
    """Набор моделей для обнаружения аномалий"""

    def __init__(self, random_state: int = DEFAULT_RANDOM_STATE):
        self.random_state = random_state
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

        # Модели
        self.isolation_forest = None
        self.dbscan = None
        self.pca = None

        # Результаты
        self.isolation_predictions = None
        self.dbscan_labels = None
        self.pca_components = None

    def train_isolation_forest(
        self,
        X: np.ndarray,
        contamination: float = DEFAULT_CONTAMINATION,
        n_estimators: int = 100
    ) -> np.ndarray:
        """
        Обучить модель Isolation Forest

        Args:
            X: Матрица признаков
            contamination: Ожидаемая доля аномалий
            n_estimators: Количество деревьев

        Returns:
            Предсказания (-1 для аномалий, 1 для нормальных)
        """
        self.logger.info(f"Обучение Isolation Forest (contamination={contamination})")

        self.isolation_forest = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )

        # Обучаем и получаем предсказания
        self.isolation_predictions = self.isolation_forest.fit_predict(X)

        # Вычисляем anomaly scores
        anomaly_scores = self.isolation_forest.decision_function(X)

        n_anomalies = np.sum(self.isolation_predictions == -1)
        anomaly_ratio = n_anomalies / len(X)

        self.logger.info(
            f"Isolation Forest: найдено {n_anomalies} аномалий ({anomaly_ratio:.2%})"
        )

        return self.isolation_predictions

    def train_dbscan(
        self,
        X: np.ndarray,
        eps: float = 0.5,
        min_samples: int = 5
    ) -> np.ndarray:
        """
        Обучить модель DBSCAN

        Args:
            X: Матрица признаков
            eps: Радиус соседства
            min_samples: Минимальное количество образцов в кластере

        Returns:
            Метки кластеров (-1 для аномалий)
        """
        self.logger.info(f"Обучение DBSCAN (eps={eps}, min_samples={min_samples})")

        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        self.dbscan_labels = self.dbscan.fit_predict(X)

        # Анализируем результаты кластеризации
        n_clusters = len(set(self.dbscan_labels)) - (1 if -1 in self.dbscan_labels else 0)
        n_noise = list(self.dbscan_labels).count(-1)
        noise_ratio = n_noise / len(X)

        self.logger.info(
            f"DBSCAN: найдено {n_clusters} кластеров, {n_noise} аномалий ({noise_ratio:.2%})"
        )

        # Вычисляем silhouette score если есть кластеры
        if n_clusters > 1:
            # Для silhouette score нужно исключить шумовые точки
            mask = self.dbscan_labels != -1
            if np.sum(mask) > 1:
                silhouette_avg = silhouette_score(X[mask], self.dbscan_labels[mask])
                self.logger.info(f"DBSCAN Silhouette Score: {silhouette_avg:.3f}")

        return self.dbscan_labels

    def train_pca(self, X: np.ndarray, n_components: int = 2) -> np.ndarray:
        """
        Обучить PCA для снижения размерности

        Args:
            X: Матрица признаков
            n_components: Количество главных компонент

        Returns:
            Трансформированные данные
        """
        self.logger.info(f"Обучение PCA (n_components={n_components})")

        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        self.pca_components = self.pca.fit_transform(X)

        # Анализируем объясненную дисперсию
        explained_variance_ratio = self.pca.explained_variance_ratio_
        total_variance = np.sum(explained_variance_ratio)

        self.logger.info(
            f"PCA: объяснено {total_variance:.2%} дисперсии "
            f"({explained_variance_ratio[0]:.2%} + {explained_variance_ratio[1]:.2%})"
        )

        return self.pca_components

    def get_feature_importance_isolation_forest(self, feature_names: List[str]) -> Dict[str, float]:
        """
        Вычислить важность признаков для Isolation Forest

        Args:
            feature_names: Названия признаков

        Returns:
            Словарь с важностью признаков
        """
        if self.isolation_forest is None:
            raise ValueError("Isolation Forest не обучен")

        # Для Isolation Forest важность вычисляется как средняя глубина разбиения
        # по каждому признаку во всех деревьях
        importances = []

        for estimator in self.isolation_forest.estimators_:
            # Получаем важность признаков для каждого дерева
            tree_importances = estimator.tree_.compute_feature_importances(normalize=False)
            importances.append(tree_importances)

        # Усредняем по всем деревьям
        mean_importances = np.mean(importances, axis=0)

        # Нормализуем
        if np.sum(mean_importances) > 0:
            mean_importances = mean_importances / np.sum(mean_importances)

        # Создаем словарь с названиями признаков
        feature_importance = dict(zip(feature_names, mean_importances))

        return feature_importance

    def get_pca_feature_contribution(self, feature_names: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Вычислить вклад каждого признака в главные компоненты PCA

        Args:
            feature_names: Названия признаков

        Returns:
            Словарь с вкладом признаков в каждую компоненту
        """
        if self.pca is None:
            raise ValueError("PCA не обучен")

        components_contribution = {}

        for i, component in enumerate(self.pca.components_):
            component_name = f'PC{i+1}'

            # Вклад каждого признака в компоненту
            contributions = dict(zip(feature_names, np.abs(component)))

            components_contribution[component_name] = contributions

        return components_contribution


class AnomalyModelTrainer:
    """Основной класс для обучения моделей аномалий"""

    def __init__(self, models_path: Optional[Path] = None):
        self.models_path = Path(models_path) if models_path else settings.models_path / "anomaly_detection"
        self.models_path.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

        # Компоненты
        self.preprocessor = FeaturePreprocessor()
        self.models = AnomalyDetectionModels()

        # Данные
        self.training_data = None
        self.feature_names = None

    async def load_training_data(
        self,
        equipment_ids: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Загрузить данные для обучения из базы данных

        Args:
            equipment_ids: Список ID оборудования (если None, загружаем все)
            limit: Максимальное количество записей

        Returns:
            DataFrame с признаками
        """
        self.logger.info("Загрузка данных для обучения из базы данных")

        async with get_async_session() as session:
            # Строим запрос
            query = select(Feature).join(RawSignal).join(Equipment)

            # Фильтруем по оборудованию если указано
            if equipment_ids:
                query = query.where(Equipment.equipment_id.in_(equipment_ids))

            # Ограничиваем количество записей
            if limit:
                query = query.limit(limit)

            # Сортируем по времени
            query = query.order_by(Feature.window_start.desc())

            # Выполняем запрос
            result = await session.execute(query)
            features = result.scalars().all()

            if len(features) < MIN_SAMPLES_FOR_TRAINING:
                raise InsufficientDataError(
                    f"Недостаточно данных для обучения: {len(features)} < {MIN_SAMPLES_FOR_TRAINING}"
                )

            # Преобразуем в DataFrame
            features_data = []
            for feature in features:
                feature_dict = {
                    'id': feature.id,
                    'equipment_id': feature.raw_signal.equipment_id,
                    'window_start': feature.window_start,
                    'window_end': feature.window_end,

                    # Статистические признаки фазы A
                    'rms_a': feature.rms_a,
                    'crest_a': feature.crest_a,
                    'kurt_a': feature.kurt_a,
                    'skew_a': feature.skew_a,
                    'mean_a': feature.mean_a,
                    'std_a': feature.std_a,
                    'min_a': feature.min_a,
                    'max_a': feature.max_a,

                    # Статистические признаки фазы B
                    'rms_b': feature.rms_b,
                    'crest_b': feature.crest_b,
                    'kurt_b': feature.kurt_b,
                    'skew_b': feature.skew_b,
                    'mean_b': feature.mean_b,
                    'std_b': feature.std_b,
                    'min_b': feature.min_b,
                    'max_b': feature.max_b,

                    # Статистические признаки фазы C
                    'rms_c': feature.rms_c,
                    'crest_c': feature.crest_c,
                    'kurt_c': feature.kurt_c,
                    'skew_c': feature.skew_c,
                    'mean_c': feature.mean_c,
                    'std_c': feature.std_c,
                    'min_c': feature.min_c,
                    'max_c': feature.max_c,
                }
                features_data.append(feature_dict)

            df = pd.DataFrame(features_data)

            self.logger.info(
                f"Загружено {len(df)} записей признаков для обучения"
            )

            return df

    async def train_models(
        self,
        equipment_ids: Optional[List[str]] = None,
        contamination: float = DEFAULT_CONTAMINATION,
        save_visualizations: bool = True
    ) -> Dict:
        """
        Обучить все модели аномалий

        Args:
            equipment_ids: Список ID оборудования
            contamination: Ожидаемая доля аномалий
            save_visualizations: Сохранять ли визуализации

        Returns:
            Словарь с результатами обучения
        """
        self.logger.info("Начинаем обучение моделей аномалий")

        # Загружаем данные
        training_df = await self.load_training_data(equipment_ids)
        self.training_data = training_df

        # Подготавливаем признаки
        X, feature_names = self.preprocessor.prepare_features_for_training(training_df)
        self.feature_names = feature_names

        # Обучаем модели
        results = {}

        # 1. Isolation Forest
        isolation_predictions = self.models.train_isolation_forest(X, contamination)
        results['isolation_forest'] = {
            'n_anomalies': int(np.sum(isolation_predictions == -1)),
            'anomaly_ratio': float(np.sum(isolation_predictions == -1) / len(X))
        }

        # 2. DBSCAN
        dbscan_labels = self.models.train_dbscan(X)
        results['dbscan'] = {
            'n_clusters': int(len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)),
            'n_anomalies': int(list(dbscan_labels).count(-1)),
            'anomaly_ratio': float(list(dbscan_labels).count(-1) / len(X))
        }

        # 3. PCA для визуализации
        pca_components = self.models.train_pca(X, n_components=2)
        results['pca'] = {
            'explained_variance_ratio': self.models.pca.explained_variance_ratio_.tolist(),
            'total_variance_explained': float(np.sum(self.models.pca.explained_variance_ratio_))
        }

        # Анализ важности признаков
        feature_importance = self.analyze_feature_importance()
        results['feature_importance'] = feature_importance

        # Сохраняем модели
        model_version = self.save_models()
        results['model_version'] = model_version

        # Создаем визуализации
        if save_visualizations:
            viz_paths = self.create_visualizations(X, pca_components)
            results['visualizations'] = viz_paths

        self.logger.info("Обучение моделей завершено успешно")

        return results

    def analyze_feature_importance(self, top_n: int = 10) -> Dict:
        """
        Анализ важности признаков

        Args:
            top_n: Количество топ признаков для вывода

        Returns:
            Словарь с анализом важности
        """
        analysis = {}

        # Важность для Isolation Forest
        if self.models.isolation_forest is not None:
            isolation_importance = self.models.get_feature_importance_isolation_forest(
                self.feature_names
            )

            # Сортируем по важности
            sorted_importance = sorted(
                isolation_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )

            analysis['isolation_forest'] = {
                'top_features': sorted_importance[:top_n],
                'all_features': isolation_importance
            }

            self.logger.info("Топ-5 признаков для Isolation Forest:")
            for name, importance in sorted_importance[:5]:
                self.logger.info(f"  {name}: {importance:.4f}")

        # Вклад в главные компоненты PCA
        if self.models.pca is not None:
            pca_contributions = self.models.get_pca_feature_contribution(self.feature_names)
            analysis['pca_contributions'] = pca_contributions

            # Находим признаки с наибольшим вкладом в первую компоненту
            pc1_contributions = sorted(
                pca_contributions['PC1'].items(),
                key=lambda x: x[1],
                reverse=True
            )

            self.logger.info("Топ-5 признаков для PC1:")
            for name, contribution in pc1_contributions[:5]:
                self.logger.info(f"  {name}: {contribution:.4f}")

        return analysis

    def create_visualizations(self, X: np.ndarray, pca_components: np.ndarray) -> Dict[str, str]:
        """
        Создать визуализации результатов

        Args:
            X: Исходные признаки
            pca_components: PCA компоненты

        Returns:
            Словарь с путями к файлам визуализаций
        """
        viz_dir = self.models_path / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        paths = {}

        # Настройка стиля
        plt.style.use('seaborn-v0_8')

        # 1. PCA scatter plot с результатами Isolation Forest
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # PCA с Isolation Forest
        colors_if = ['red' if pred == -1 else 'blue' for pred in self.models.isolation_predictions]
        axes[0].scatter(pca_components[:, 0], pca_components[:, 1], c=colors_if, alpha=0.6)
        axes[0].set_xlabel(f'PC1 ({self.models.pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[0].set_ylabel(f'PC2 ({self.models.pca.explained_variance_ratio_[1]:.1%} variance)')
        axes[0].set_title('PCA + Isolation Forest\n(красные = аномалии)')
        axes[0].grid(True, alpha=0.3)

        # PCA с DBSCAN
        unique_labels = set(self.models.dbscan_labels)
        colors_db = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        for k, col in zip(unique_labels, colors_db):
            if k == -1:
                # Аномалии черным цветом
                col = [0, 0, 0, 1]

            class_member_mask = (self.models.dbscan_labels == k)
            xy = pca_components[class_member_mask]

            axes[1].scatter(xy[:, 0], xy[:, 1], c=[col], alpha=0.6,
                          label=f'Cluster {k}' if k != -1 else 'Anomalies')

        axes[1].set_xlabel(f'PC1 ({self.models.pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[1].set_ylabel(f'PC2 ({self.models.pca.explained_variance_ratio_[1]:.1%} variance)')
        axes[1].set_title('PCA + DBSCAN')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        pca_path = viz_dir / f"pca_analysis_{timestamp}.png"
        plt.savefig(pca_path, dpi=300, bbox_inches='tight')
        plt.close()
        paths['pca_analysis'] = str(pca_path)

        # 2. Feature importance для Isolation Forest
        if hasattr(self, 'feature_names') and self.models.isolation_forest is not None:
            importance_dict = self.models.get_feature_importance_isolation_forest(self.feature_names)

            # Сортируем и берем топ-15
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:15]

            fig, ax = plt.subplots(figsize=(10, 8))

            features, importances = zip(*sorted_features)
            y_pos = np.arange(len(features))

            bars = ax.barh(y_pos, importances)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.invert_yaxis()
            ax.set_xlabel('Feature Importance')
            ax.set_title('Top 15 Features - Isolation Forest Importance')
            ax.grid(True, alpha=0.3)

            # Добавляем значения на барах
            for i, (bar, importance) in enumerate(zip(bars, importances)):
                ax.text(importance + 0.001, bar.get_y() + bar.get_height()/2,
                       f'{importance:.3f}', va='center', fontsize=8)

            plt.tight_layout()
            importance_path = viz_dir / f"feature_importance_{timestamp}.png"
            plt.savefig(importance_path, dpi=300, bbox_inches='tight')
            plt.close()
            paths['feature_importance'] = str(importance_path)

        # 3. Распределение аномалий по времени
        if self.training_data is not None:
            fig, ax = plt.subplots(figsize=(12, 6))

            # Добавляем метки аномалий к данным
            df_viz = self.training_data.copy()
            df_viz['is_anomaly_if'] = self.models.isolation_predictions == -1
            df_viz['is_anomaly_db'] = self.models.dbscan_labels == -1

            # Группируем по часам и считаем аномалии
            df_viz['hour'] = pd.to_datetime(df_viz['window_start']).dt.floor('H')

            hourly_stats = df_viz.groupby('hour').agg({
                'is_anomaly_if': ['sum', 'count'],
                'is_anomaly_db': 'sum'
            }).reset_index()

            hourly_stats.columns = ['hour', 'anomalies_if', 'total_samples', 'anomalies_db']
            hourly_stats['anomaly_rate_if'] = hourly_stats['anomalies_if'] / hourly_stats['total_samples']
            hourly_stats['anomaly_rate_db'] = hourly_stats['anomalies_db'] / hourly_stats['total_samples']

            ax.plot(hourly_stats['hour'], hourly_stats['anomaly_rate_if'],
                   label='Isolation Forest', marker='o', alpha=0.7)
            ax.plot(hourly_stats['hour'], hourly_stats['anomaly_rate_db'],
                   label='DBSCAN', marker='s', alpha=0.7)

            ax.set_xlabel('Время')
            ax.set_ylabel('Доля аномалий')
            ax.set_title('Распределение аномалий по времени')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)

            plt.tight_layout()
            timeline_path = viz_dir / f"anomaly_timeline_{timestamp}.png"
            plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
            plt.close()
            paths['anomaly_timeline'] = str(timeline_path)

        self.logger.info(f"Создано {len(paths)} визуализаций в {viz_dir}")

        return paths

    def save_models(self) -> str:
        """
        Сохранить обученные модели

        Returns:
            Версия модели
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"v1.0.0_{timestamp}"

        version_dir = self.models_path / version
        version_dir.mkdir(exist_ok=True)

        # Сохраняем модели
        models_to_save = {
            'isolation_forest': self.models.isolation_forest,
            'dbscan': self.models.dbscan,
            'pca': self.models.pca,
            'preprocessor': self.preprocessor
        }

        for model_name, model in models_to_save.items():
            if model is not None:
                model_path = version_dir / f"{model_name}.pkl"
                joblib.dump(model, model_path, compress=3)
                self.logger.debug(f"Сохранена модель {model_name} в {model_path}")

        # Сохраняем метаданные
        metadata = {
            'version': version,
            'created_at': datetime.now().isoformat(),
            'model_type': 'anomaly_detection',
            'models': {
                'isolation_forest': {
                    'contamination': getattr(self.models.isolation_forest, 'contamination', None),
                    'n_estimators': getattr(self.models.isolation_forest, 'n_estimators', None)
                },
                'dbscan': {
                    'eps': getattr(self.models.dbscan, 'eps', None),
                    'min_samples': getattr(self.models.dbscan, 'min_samples', None)
                },
                'pca': {
                    'n_components': getattr(self.models.pca, 'n_components', None),
                    'explained_variance_ratio': getattr(self.models.pca, 'explained_variance_ratio_', []).tolist() if self.models.pca else None
                }
            },
            'feature_names': self.feature_names,
            'training_samples': len(self.training_data) if self.training_data is not None else 0
        }

        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Создаем симлинк на latest
        latest_path = self.models_path / "latest"
        if latest_path.exists():
            if latest_path.is_symlink():
                latest_path.unlink()
            else:
                import shutil
                shutil.rmtree(latest_path)

        try:
            latest_path.symlink_to(version, target_is_directory=True)
        except OSError:
            # На Windows может не работать, копируем вместо симлинка
            import shutil
            shutil.copytree(version_dir, latest_path)

        self.logger.info(f"Модели сохранены в версии {version}")

        return version


# CLI функции

async def train_anomaly_models(
    equipment_ids: Optional[List[str]] = None,
    contamination: float = DEFAULT_CONTAMINATION,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Обучить модели аномалий

    Args:
        equipment_ids: Список ID оборудования
        contamination: Ожидаемая доля аномалий
        output_dir: Директория для сохранения моделей

    Returns:
        Результаты обучения
    """
    output_path = Path(output_dir) if output_dir else None
    trainer = AnomalyModelTrainer(models_path=output_path)

    results = await trainer.train_models(
        equipment_ids=equipment_ids,
        contamination=contamination,
        save_visualizations=True
    )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Обучение моделей аномалий для диагностики двигателей")
    parser.add_argument("--equipment-ids", nargs="+", help="Список ID оборудования")
    parser.add_argument("--contamination", type=float, default=DEFAULT_CONTAMINATION,
                       help="Ожидаемая доля аномалий")
    parser.add_argument("--output-dir", help="Директория для сохранения моделей")
    parser.add_argument("--verbose", "-v", action="store_true", help="Подробный вывод")

    args = parser.parse_args()

    # Настраиваем логирование
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)

    async def main():
        try:
            logger.info("Начинаем обучение моделей аномалий")

            results = await train_anomaly_models(
                equipment_ids=args.equipment_ids,
                contamination=args.contamination,
                output_dir=args.output_dir
            )

            print(f"\n✓ Обучение завершено успешно!")
            print(f"  - Версия модели: {results['model_version']}")
            print(f"  - Isolation Forest: {results['isolation_forest']['n_anomalies']} аномалий "
                  f"({results['isolation_forest']['anomaly_ratio']:.1%})")
            print(f"  - DBSCAN: {results['dbscan']['n_clusters']} кластеров, "
                  f"{results['dbscan']['n_anomalies']} аномалий")
            print(f"  - PCA: объяснено {results['pca']['total_variance_explained']:.1%} дисперсии")

            if 'visualizations' in results:
                print(f"  - Создано визуализаций: {len(results['visualizations'])}")

        except Exception as e:
            logger.error(f"Ошибка обучения моделей: {e}")
            print(f"✗ Ошибка: {e}")
            return False

        return True

    import asyncio
    success = asyncio.run(main())
    exit(0 if success else 1)
