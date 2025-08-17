# Обучение моделей аномалий (streaming / stats baseline)

import json
import pickle
import warnings
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.settings import get_settings
from src.database.connection import get_async_session
from src.database.models import Feature, Equipment, RawSignal
from src.utils.logger import get_logger
try:  # river опционален
    from src.ml.incremental import HalfSpaceTreesIncremental
    RIVER_OK = True
except Exception:  # pragma: no cover
    RIVER_OK = False

# Настройки
settings = get_settings()
logger = get_logger(__name__)

# Константы
DEFAULT_CONTAMINATION = 0.1  # Ожидаемая доля аномалий
DEFAULT_RANDOM_STATE = 42
MIN_SAMPLES_FOR_TRAINING = 100


def retrain_stream_or_stats_minimal(prefer_stream: bool = True) -> str:
    """Минимальное переобучение новой *поточной* (stream) или статистической (stats) модели аномалий.

    Логика:
      1. Загружаем до 5000 последних записей признаков (как и в IF минималке)
      2. Пытаемся инициализировать StreamingHalfSpaceTreesAdapter (river)
         - Если river недоступен или данных мало (< 30), используем StatsQuantileBaseline
      3. Для stream: проходим по объектам, собираем предварительные score (до learn_one) => вычисляем порог как квантиль (0.98) или fallback 0.7
         Сохраняем состояние модели в stream_state.pkl (joblib)
      4. Для stats: кормим baseline.update, вычисляем median/MAD, сохраняем state в stats_state.json
      5. manifest.json перезаписываем полями: model_type, threshold, version, n_samples, created_at, quantile(если stream)

    Возвращает строку версии (UTC timestamp).
    """
    import asyncio, json, joblib
    from datetime import datetime, UTC
    import numpy as _np
    from pathlib import Path
    st = get_settings()
    models_dir = st.models_path / 'anomaly_detection' / 'latest'
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Загрузка признаков (аналогично функции выше)
    async def _load():
        async with get_async_session() as session:
            res = await session.execute(select(Feature))
            feats = res.scalars().all()
            return feats
    feats = asyncio.run(_load())  # type: ignore
    if not feats:
        raise InsufficientDataError('Нет признаков для переобучения (stream/stats)')

    # Берём ограниченный набор
    feats = feats[:5000]
    # Формируем словари только с базовыми статистическими признаками (rms/crest/kurt/skew + mean/std/min/max для фаз)
    stat_cols = [
        'rms_a','rms_b','rms_c','crest_a','crest_b','crest_c','kurt_a','kurt_b','kurt_c',
        'skew_a','skew_b','skew_c','mean_a','mean_b','mean_c','std_a','std_b','std_c',
        'min_a','min_b','min_c','max_a','max_b','max_c']
    rows: list[dict[str,float]] = []
    for f in feats:
        d = {}
        for c in stat_cols:
            v = getattr(f, c, None)
            if v is not None:
                try:
                    d[c] = float(v)
                except Exception:
                    pass
        if d:
            rows.append(d)

    if len(rows) < 5:  # слишком мало для чего-либо осмысленного
        # fallback сразу на baseline c заглушкой
        from src.ml.incremental import StatsQuantileBaseline
        baseline = StatsQuantileBaseline()
        for r in rows:
            baseline.update(r)
        arr = _np.array(baseline.values) if baseline.values else _np.array([0.0])
        med = float(_np.median(arr))
        mad = float(_np.median(_np.abs(arr - med))) + 1e-9
        version = datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')
        (models_dir / 'stats_state.json').write_text(json.dumps({
            'z_threshold': baseline.z_threshold,
            'n_samples': len(arr),
            'median': med,
            'mad': mad
        }, indent=2, ensure_ascii=False), encoding='utf-8')
        manifest = {
            'model_type': 'stats',
            'threshold': baseline.z_threshold,
            'version': version,
            'n_samples': len(arr),
            'created_at': datetime.now(UTC).isoformat()
        }
        (models_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
        logger.info(f"[retrain] Stats baseline (few samples) версия {version}")
        return version

    # 2. Попытка stream
    used_stream = False
    threshold = 0.7
    quantile = None
    outlier_ratio = None
    version = datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')
    try:
        if prefer_stream:
            from src.ml.incremental import StreamingHalfSpaceTreesAdapter  # type: ignore
            adapter = StreamingHalfSpaceTreesAdapter(threshold=threshold)
            scores: list[float] = []
            # Собираем объединённое множество всех используемых ключей для согласованности (опционально не обязательно)
            for r in rows:
                # score перед learn
                s = adapter.model.score_one(r)  # type: ignore[attr-defined]
                scores.append(float(s))
                adapter.model.learn_one(r)  # type: ignore[attr-defined]
            if len(scores) >= 30:
                quantile = float(_np.quantile(_np.array(scores), 0.98))
                # Порог = max(default, quantile) чтобы не занижать
                threshold = max(0.7, quantile)
                adapter.threshold = threshold
            if scores:
                arr_scores = _np.array(scores)
                outlier_ratio = float((arr_scores > threshold).mean()) if threshold > 0 else 0.0
            # Сохраняем состояние
            joblib.dump({'model': adapter.model}, models_dir / 'stream_state.pkl', compress=3)
            manifest = {
                'model_type': 'stream',
                'threshold': threshold,
                'version': version,
                'n_samples': len(rows),
                'created_at': datetime.now(UTC).isoformat(),
                'scores_p98': quantile,
                'outlier_ratio': outlier_ratio,
            }
            (models_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
            logger.info(f"[retrain] Stream HalfSpaceTrees обновлена версия {version} (threshold={threshold:.4f})")
            used_stream = True
    except Exception as e:  # pragma: no cover
        logger.warning(f"[retrain] stream недоступен: {e}; fallback -> stats")

    if used_stream:
        return version

    # 3. Fallback stats baseline
    from src.ml.incremental import StatsQuantileBaseline
    baseline = StatsQuantileBaseline()
    for r in rows:
        baseline.update(r)
    arr = _np.array(baseline.values)
    med = float(_np.median(arr))
    mad = float(_np.median(_np.abs(arr - med))) + 1e-9
    (models_dir / 'stats_state.json').write_text(json.dumps({
        'z_threshold': baseline.z_threshold,
        'n_samples': len(arr),
        'median': med,
        'mad': mad
    }, indent=2, ensure_ascii=False), encoding='utf-8')
    # Для stats используем оценку outlier_ratio как долю значений с z > threshold
    try:
        z_scores = _np.abs((arr - med) / (mad if mad != 0 else 1.0))
        outlier_ratio = float((z_scores > baseline.z_threshold).mean())
    except Exception:
        outlier_ratio = None
    manifest = {
        'model_type': 'stats',
        'threshold': baseline.z_threshold,
        'version': version,
        'n_samples': len(arr),
        'created_at': datetime.now(UTC).isoformat(),
        'median': med,
        'mad': mad,
        'outlier_ratio': outlier_ratio
    }
    (models_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    logger.info(f"[retrain] Stats baseline обновлена версия {version} (z>{baseline.z_threshold})")
    return version

def load_latest_models() -> Dict[str, object]:
    """Заглушка загрузки legacy моделей (возвращает пустой словарь)."""
    return {}


class AnomalyModelError(Exception):
    # Базовое исключение моделей аномалий
    pass


class InsufficientDataError(AnomalyModelError):
    # Недостаточно данных
    pass


async def train_isolation_forest(*_args, **_kwargs):  # type: ignore
    """Удалённая функция (raise)."""
    raise RuntimeError("train_isolation_forest удалён; используйте потоковую anomaly модель.")


class FeaturePreprocessor:
    # Предобработка признаков

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
    # Подготовка признаков для обучения
    # Замечание: первоначально проверка MIN_SAMPLES выполнялась до базовой валидации,
    # что приводило к выбросу InsufficientDataError вместо ожидаемого тестом ValueError,
    # когда отсутствуют нужные признаки. Сначала определим доступные признаки, затем проверим размер.

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

        # Теперь выполняем глобальную проверку достаточности данных после проверки наличия признаков
        if len(features_df) < MIN_SAMPLES_FOR_TRAINING:
            raise InsufficientDataError(
                f"Недостаточно данных для обучения: {len(features_df)} < {MIN_SAMPLES_FOR_TRAINING}"
            )

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
        # Обработка пропусков
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
        # Отбор по дисперсии
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
        # Нормализация
        # Используем RobustScaler для устойчивости к выбросам
        self.feature_scaler = RobustScaler()
        X_scaled = self.feature_scaler.fit_transform(X)

        return X_scaled

    def transform_features(self, features_df: pd.DataFrame) -> np.ndarray:
        # Трансформация новых данных
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
    # Набор моделей аномалий

    def __init__(self, random_state: int = DEFAULT_RANDOM_STATE):
        self.random_state = random_state
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
    # Модели
        self.dbscan = None
        self.pca = None
        self.incremental_adapter = None  # HalfSpaceTreesIncremental

        # Результаты
        self.dbscan_labels = None
        self.pca_components = None

    # Legacy IsolationForest удалён.

    def train_incremental(self, X: np.ndarray, feature_names: List[str]):
        """Инкрементальная (поточная) модель: HalfSpaceTrees если доступен river.

        Если библиотека river недоступна, оставляем предупреждение и не инициализируем адаптер.
        """
        if RIVER_OK:
            try:
                adapter = HalfSpaceTreesIncremental()
                for row in X:
                    adapter.learn_one({fn: float(v) for fn, v in zip(feature_names, row)})
                self.incremental_adapter = adapter
                self.logger.info("Инкрементальная модель HalfSpaceTrees обучена")
                return
            except Exception as e:  # pragma: no cover
                self.logger.warning(f"HalfSpaceTrees недоступен: {e}")
        else:
            self.logger.warning("river недоступен: потоковая модель не инициализирована")

    def train_dbscan(
        self,
        X: np.ndarray,
        eps: float = 0.5,
        min_samples: int = 5
    ) -> np.ndarray:
        # Обучить DBSCAN, вернуть метки (-1 аномалии)
        self.logger.info(f"Обучение DBSCAN (eps={eps}, min_samples={min_samples})")

        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        self.dbscan_labels = self.dbscan.fit_predict(X)

        def _report(labels, attempt_tag: str = "primary"):
            n_clusters_local = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_local = list(labels).count(-1)
            noise_ratio_local = n_noise_local / len(labels)
            self.logger.info(
                f"DBSCAN({attempt_tag}): найдено {n_clusters_local} кластеров, {n_noise_local} аномалий ({noise_ratio_local:.2%})"
            )
            return n_clusters_local

        n_clusters = _report(self.dbscan_labels)
        # Fallback: если ни одного кластера (всё шум) – ослабляем параметры и пробуем ещё раз.
        if n_clusters == 0:
            relaxed_eps = eps * 1.5
            relaxed_min_samples = max(2, int(min_samples * 0.5))
            try:
                self.logger.info(
                    f"DBSCAN fallback: пересобираем с eps={relaxed_eps}, min_samples={relaxed_min_samples}"
                )
                self.dbscan = DBSCAN(eps=relaxed_eps, min_samples=relaxed_min_samples, n_jobs=-1)
                self.dbscan_labels = self.dbscan.fit_predict(X)
                n_clusters = _report(self.dbscan_labels, attempt_tag="fallback")
            except Exception as e:
                self.logger.warning(f"DBSCAN fallback ошибка: {e}")

        # Вычисляем silhouette score если есть кластеры
        if n_clusters > 1:
            # Для silhouette score нужно исключить шумовые точки
            mask = self.dbscan_labels != -1
            if np.sum(mask) > 1:
                silhouette_avg = silhouette_score(X[mask], self.dbscan_labels[mask])
                self.logger.info(f"DBSCAN Silhouette Score: {silhouette_avg:.3f}")

        return self.dbscan_labels

    def train_pca(self, X: np.ndarray, n_components: int = 2) -> np.ndarray:
        # Обучить PCA и вернуть компоненты
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

    # Метод важности IsolationForest удалён.

    def get_pca_feature_contribution(self, feature_names: List[str]) -> Dict[str, Dict[str, float]]:
        # Вклад признаков в компоненты PCA
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
    # Управление процессом обучения моделей аномалий

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
    # Загрузить данные признаков из БД
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

            # Стратифицированное/ограниченное семплирование для масштабируемости
            from src.config.settings import get_settings
            settings_local = get_settings()
            if settings_local.ANOMALY_TRAIN_SAMPLE_SIZE:
                target_size = settings_local.ANOMALY_TRAIN_SAMPLE_SIZE
                if len(df) > target_size:
                    if settings_local.ANOMALY_TRAIN_STRATIFIED and 'equipment_id' in df.columns:
                        # Пропорционально количеству записей на оборудование
                        sampled_frames = []
                        remaining = target_size
                        equip_groups = df.groupby('equipment_id')
                        total = len(df)
                        for eq, group in equip_groups:
                            # Минимум 1, пропорционально размеру группы
                            take = max(1, int(len(group) / total * target_size))
                            if take > len(group):
                                take = len(group)
                            sampled_frames.append(group.sample(n=take, random_state=42))
                            remaining -= take
                        if remaining > 0:
                            # добираем случайными из оставшихся
                            rest = pd.concat(sampled_frames)
                            missing = target_size - len(rest)
                            if missing > 0:
                                others = df.drop(rest.index)
                                if len(others) > 0:
                                    sampled_frames.append(others.sample(n=min(missing, len(others)), random_state=42))
                        df = pd.concat(sampled_frames).sample(frac=1, random_state=42).reset_index(drop=True)
                    else:
                        df = df.sample(n=target_size, random_state=42).reset_index(drop=True)

                    self.logger.info(f"Применено семплирование: использовано {len(df)} записей из {total}")

            return df

    async def train_models(
        self,
        equipment_ids: Optional[List[str]] = None,
        contamination: float = DEFAULT_CONTAMINATION,
        save_visualizations: bool = True
    ) -> Dict:
    # Обучить все модели аномалий
        self.logger.info("Начинаем обучение моделей аномалий")

        # Загружаем данные
        training_df = await self.load_training_data(equipment_ids)
        self.training_data = training_df

        # Подготавливаем признаки
        X, feature_names = self.preprocessor.prepare_features_for_training(training_df)
        self.feature_names = feature_names

        # Обучаем модели
        results = {}

        # 1. Incremental / streaming модель
        try:
            self.models.train_incremental(X, feature_names)
            results['incremental_model'] = 'half_space_trees' if RIVER_OK else 'none'
        except Exception as e:  # pragma: no cover
            self.logger.warning(f"Инкрементальная модель не обучена: {e}")

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
        analysis: Dict[str, Dict] = {}
        if self.models.pca is not None:
            pca_contributions = self.models.get_pca_feature_contribution(self.feature_names)
            analysis['pca_contributions'] = pca_contributions
            pc1_contributions = sorted(pca_contributions['PC1'].items(), key=lambda x: x[1], reverse=True)
            self.logger.info("Топ-5 признаков для PC1:")
            for name, contribution in pc1_contributions[:5]:
                self.logger.info(f"  {name}: {contribution:.4f}")
        return analysis

    def create_visualizations(self, X: np.ndarray, pca_components: np.ndarray) -> Dict[str, str]:
        # Построить визуализации результатов
        viz_dir = self.models_path / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        paths = {}

        # Настройка стиля
        plt.style.use('seaborn-v0_8')

        # 1. PCA scatter plot: DBSCAN кластеры
        fig, ax = plt.subplots(figsize=(7, 6))
        unique_labels = set(self.models.dbscan_labels)
        colors_db = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        for k, col in zip(unique_labels, colors_db):
            if k == -1:
                col = [0, 0, 0, 1]
            class_member_mask = (self.models.dbscan_labels == k)
            xy = pca_components[class_member_mask]
            ax.scatter(xy[:, 0], xy[:, 1], c=[col], alpha=0.6,
                       label=f'Cluster {k}' if k != -1 else 'Anomalies')
        ax.set_xlabel(f'PC1 ({self.models.pca.explained_variance_ratio_[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({self.models.pca.explained_variance_ratio_[1]:.1%} variance)')
        ax.set_title('PCA + DBSCAN')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        pca_path = viz_dir / f"pca_analysis_{timestamp}.png"
        plt.savefig(pca_path, dpi=300, bbox_inches='tight')
        plt.close()
        paths['pca_analysis'] = str(pca_path)

        # 2. Распределение аномалий (DBSCAN) по времени
        if self.training_data is not None:
            fig, ax = plt.subplots(figsize=(12, 6))

            # Добавляем метки аномалий к данным
            df_viz = self.training_data.copy()
            df_viz['is_anomaly_db'] = self.models.dbscan_labels == -1

            # Группируем по часам и считаем аномалии
            df_viz['hour'] = pd.to_datetime(df_viz['window_start']).dt.floor('h')

            hourly_stats = df_viz.groupby('hour').agg({
                'is_anomaly_db': 'sum'
            }).reset_index()
            hourly_stats.columns = ['hour', 'anomalies_db']
            total_samples = df_viz.groupby('hour').size().reindex(hourly_stats['hour']).values
            anomaly_rate_db = hourly_stats['anomalies_db'] / total_samples
            ax.plot(hourly_stats['hour'], anomaly_rate_db, label='DBSCAN', marker='s', alpha=0.7)

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
        # Сохранить модели и вернуть версию
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"v1.0.0_{timestamp}"

        version_dir = self.models_path / version
        version_dir.mkdir(exist_ok=True)

        # Сохраняем модели
        models_to_save = {
            'dbscan': self.models.dbscan,
            'pca': self.models.pca,
            'preprocessor': self.preprocessor,
            'incremental': self.models.incremental_adapter
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
    # Обучить модели аномалий (все)
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

            logger.info("Обучение завершено успешно")
            logger.info(f"Версия модели: {results['model_version']}")
            logger.info(
                f"DBSCAN: {results['dbscan']['n_clusters']} кластеров, "
                f"{results['dbscan']['n_anomalies']} аномалий"
            )
            logger.info(
                f"PCA: объяснено {results['pca']['total_variance_explained']:.1%} дисперсии"
            )

            if 'visualizations' in results:
                logger.info(f"Создано визуализаций: {len(results['visualizations'])}")

        except Exception as e:
            logger.error(f"Ошибка обучения моделей: {e}")
            # Вывод в stdout убираем: достаточно логера
            return False

        return True

    import asyncio
    success = asyncio.run(main())
    exit(0 if success else 1)
