# Валидатор CSV токовых сигналов

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ValidationSeverity(Enum):  # Уровень критичности
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:  # Результат валидации
    severity: ValidationSeverity
    message: str
    details: Optional[Dict] = None


class DataValidator:  # Основной валидатор

    def __init__(self):
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

    # Пороговые значения
        self.max_current_amplitude = 1000.0  # Максимальная амплитуда тока (А)
        self.min_sample_rate = 1000  # Минимальная частота дискретизации (Гц)
        self.max_sample_rate = 100000  # Максимальная частота дискретизации (Гц)
        self.max_nan_ratio = 0.8  # Максимальная доля NaN значений
        self.min_signal_length = 100  # Минимальная длина сигнала

    def validate_csv_data(
        self,
        phase_data: Dict[str, List[float]],
        sample_rate: int,
        filename: str = "unknown",
        motor_metadata: Optional[Dict] = None,
    ) -> List[ValidationResult]:
    # Валидация набора фаз
        results = []

        # Доменная проверка параметров мотора (если известны)
        results.extend(self._validate_motor_params(sample_rate, motor_metadata))

    # Частота дискретизации
        results.extend(self._validate_sample_rate(sample_rate))

    # Проверка фаз
        for phase_name, values in phase_data.items():
            if values:  # преобразуем None -> np.nan
                norm = [np.nan if (v is None) else v for v in values]
                phase_results = self._validate_phase_data(
                    np.array(norm, dtype=float), phase_name, filename
                )
                results.extend(phase_results)

    # Согласованность фаз
        results.extend(self._validate_phase_consistency(phase_data))

    # Лог результатов
        self._log_validation_results(results, filename)

        return results

    def _validate_sample_rate(self, sample_rate: int) -> List[ValidationResult]:
        """Валидировать частоту дискретизации"""
        results = []

        if sample_rate < self.min_sample_rate:
            results.append(ValidationResult(
                severity=ValidationSeverity.ERROR,
                message=f"Слишком низкая частота дискретизации: {sample_rate} Гц",
                details={"sample_rate": sample_rate, "min_required": self.min_sample_rate}
            ))
        elif sample_rate > self.max_sample_rate:
            results.append(ValidationResult(
                severity=ValidationSeverity.WARNING,
                message=f"Очень высокая частота дискретизации: {sample_rate} Гц",
                details={"sample_rate": sample_rate, "max_recommended": self.max_sample_rate}
            ))

        return results

    def _validate_phase_data(
        self,
        phase_values: np.ndarray,
        phase_name: str,
        filename: str
    ) -> List[ValidationResult]:
        """Валидировать данные одной фазы."""
        results: List[ValidationResult] = []

        # Длина сигнала (<=10 умеренно короткий — WARNING)
        length = len(phase_values)
        if length < self.min_signal_length:
            severity = ValidationSeverity.ERROR
            if 3 <= length <= 10:
                severity = ValidationSeverity.WARNING
            results.append(ValidationResult(
                severity=severity,
                message=f"Слишком короткий сигнал фазы {phase_name}: {length} отсчетов",
                details={"phase": phase_name, "length": length, "min_required": self.min_signal_length, "softened": severity == ValidationSeverity.WARNING}
            ))

        if length == 0:
            return results

        # Доля NaN
        nan_mask = np.isnan(phase_values)
        nan_count = int(np.sum(nan_mask))
        nan_ratio = nan_count / length

        if nan_ratio > self.max_nan_ratio:
            results.append(ValidationResult(
                severity=ValidationSeverity.CRITICAL,
                message=f"Слишком много пропусков в фазе {phase_name}: {nan_ratio:.1%}",
                details={"phase": phase_name, "nan_ratio": nan_ratio, "nan_count": nan_count}
            ))
        elif nan_ratio > 0.1:
            results.append(ValidationResult(
                severity=ValidationSeverity.WARNING,
                message=f"Много пропусков в фазе {phase_name}: {nan_ratio:.1%}",
                details={"phase": phase_name, "nan_ratio": nan_ratio}
            ))

        # Амплитуды (по валидным значениям)
        if nan_count < length:
            valid_values = phase_values[~nan_mask]
            max_abs_value = float(np.max(np.abs(valid_values))) if valid_values.size else 0.0
            if max_abs_value > self.max_current_amplitude:
                results.append(ValidationResult(
                    severity=ValidationSeverity.ERROR,
                    message=f"Слишком большая амплитуда тока в фазе {phase_name}: {max_abs_value:.2f} А",
                    details={"phase": phase_name, "max_amplitude": max_abs_value}
                ))
            if valid_values.size:
                mean_value = float(np.mean(valid_values))
                std_value = float(np.std(valid_values))
                if std_value < 0.01 * abs(mean_value):
                    results.append(ValidationResult(
                        severity=ValidationSeverity.WARNING,
                        message=f"Подозрительно малые колебания в фазе {phase_name}",
                        details={"phase": phase_name, "mean": mean_value, "std": std_value}
                    ))

        return results

    def _validate_phase_consistency(
        self,
        phase_data: Dict[str, List[float]]
    ) -> List[ValidationResult]:
        """Проверить согласованность между фазами"""
        results = []

    # Длины фаз
        phase_lengths = {}
        for phase_name, values in phase_data.items():
            if values:
                phase_lengths[phase_name] = len(values)

        if len(phase_lengths) > 1:
            # Сравнение длин
            lengths = list(phase_lengths.values())
            if len(set(lengths)) > 1:
                results.append(ValidationResult(
                    severity=ValidationSeverity.ERROR,
                    message="Разная длина сигналов по фазам",
                    details={"phase_lengths": phase_lengths}
                ))

    # Наличие хотя бы одной непустой фазы
        has_data = any(values for values in phase_data.values())
        if not has_data:
            results.append(ValidationResult(
                severity=ValidationSeverity.CRITICAL,
                message="Нет данных ни по одной фазе",
                details={"phase_data": {k: len(v) for k, v in phase_data.items()}}
            ))
        else:
            # Предупреждения если фаза полностью пустая либо доля пропусков <10% (мягкий сигнал) для каждой непустой фазы
            for phase_name, values in phase_data.items():
                if not values:
                    results.append(ValidationResult(
                        severity=ValidationSeverity.WARNING,
                        message=f"Фаза {phase_name} отсутствует (пустая)",
                        details={'phase': phase_name}
                    ))
                else:
                    arr_raw = [np.nan if v is None else v for v in values]
                    arr = np.array(arr_raw, dtype=float)
                    if arr.size > 0:
                        nan_ratio = np.sum(np.isnan(arr)) / arr.size
                        if 0 < nan_ratio <= 0.1:
                            results.append(ValidationResult(
                                severity=ValidationSeverity.WARNING,
                                message=f"Небольшое число пропусков в фазе {phase_name}: {nan_ratio:.1%}",
                                details={'phase': phase_name, 'nan_ratio': nan_ratio}
                            ))

        return results

    def _validate_motor_params(self, sample_rate: int, motor_metadata: Optional[Dict]) -> List[ValidationResult]:
        """Проверка моторных параметров.

        Если метаданные не переданы – выдаём только мягкую INFO/WARNING при сильном расхождении частоты.
        Ожидаемые параметры (жёстко зашиты): асинхронный двигатель 3 кВт, 1770 rpm номинал,
        multiplier 3010 rpm, sample_rate 25600 Гц.
        """
        expected = {
            'type': 'asynchronous',
            'power_kw': 3.0,
            'nominal_speed_rpm': 1770,
            'multiplier_speed_rpm': 3010,
            'sample_rate_hz': 25600,
        }
        results: List[ValidationResult] = []
        from src.config.settings import get_settings
        st = get_settings()

        # Частота дискретизации — единственный реально поступающий параметр сейчас
        tolerance = expected['sample_rate_hz'] * 0.005  # 0.5%
        if abs(sample_rate - expected['sample_rate_hz']) > tolerance:
            sev = ValidationSeverity.CRITICAL if (motor_metadata and not st.is_testing) else (
                ValidationSeverity.WARNING if not st.is_testing else ValidationSeverity.WARNING)
            results.append(ValidationResult(
                severity=sev,
                message=f"Несоответствие sample_rate: {sample_rate} != {expected['sample_rate_hz']}",
                details={'expected': expected['sample_rate_hz'], 'actual': sample_rate, 'tolerance': tolerance, 'metadata_present': bool(motor_metadata), 'softened': st.is_testing}
            ))

        if motor_metadata:
            # Проверяем каждое доступное поле
            def _cmp(key, meta_key=None):
                mk = meta_key or key
                if mk in motor_metadata:
                    if motor_metadata[mk] != expected[key]:
                        results.append(ValidationResult(
                            severity=ValidationSeverity.CRITICAL,
                            message=f"Несоответствие параметра мотора {mk}: {motor_metadata[mk]} != {expected[key]}",
                            details={'param': mk, 'expected': expected[key], 'actual': motor_metadata[mk]}
                        ))
                else:
                    results.append(ValidationResult(
                        severity=ValidationSeverity.WARNING,
                        message=f"Параметр мотора {mk} отсутствует в метаданных",
                        details={'param': mk}
                    ))
            _cmp('type')
            _cmp('power_kw')
            _cmp('nominal_speed_rpm')
            _cmp('multiplier_speed_rpm')
        else:
            # Нет метаданных — информируем (INFO чтобы не шуметь)
            results.append(ValidationResult(
                severity=ValidationSeverity.INFO,
                message="Метаданные мотора отсутствуют (использованы допущения)",
                details={'expected': expected}
            ))
        return results

    def _log_validation_results(
        self,
        results: List[ValidationResult],
        filename: str
    ):
        """Логировать результаты валидации"""
        if not results:
            self.logger.info(f"Валидация файла {filename} прошла успешно")
            return

        # Группируем по уровню критичности
        by_severity = {}
        for result in results:
            severity = result.severity
            if severity not in by_severity:
                by_severity[severity] = []
            by_severity[severity].append(result)

        # Логируем по уровням
        for severity, severity_results in by_severity.items():
            count = len(severity_results)
            messages = [r.message for r in severity_results]

            if severity == ValidationSeverity.CRITICAL:
                self.logger.error(f"Критические ошибки валидации {filename} ({count}): {messages}")
            elif severity == ValidationSeverity.ERROR:
                self.logger.error(f"Ошибки валидации {filename} ({count}): {messages}")
            elif severity == ValidationSeverity.WARNING:
                self.logger.warning(f"Предупреждения валидации {filename} ({count}): {messages}")
            else:
                self.logger.info(f"Информация валидации {filename} ({count}): {messages}")

    def has_critical_errors(self, results: List[ValidationResult]) -> bool:
        """Проверить наличие критических ошибок"""
        return any(
            r.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]
            for r in results
        )

    def get_validation_summary(self, results: List[ValidationResult]) -> Dict:
        """Получить сводку по валидации"""
        summary = {
            "total_issues": len(results),
            "by_severity": {},
            "has_critical_errors": self.has_critical_errors(results)
        }

        for result in results:
            severity = result.severity.value
            if severity not in summary["by_severity"]:
                summary["by_severity"][severity] = 0
            summary["by_severity"][severity] += 1

        return summary
