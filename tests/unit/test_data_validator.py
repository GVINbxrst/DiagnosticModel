import numpy as np
from src.data_processing.data_validator import DataValidator, ValidationSeverity


def test_validate_good_data():
    v = DataValidator()
    phase_data = {
        'R': list(np.random.normal(0,1,500)),
        'S': list(np.random.normal(0,1,500)),
        'T': list(np.random.normal(0,1,500)),
    }
    results = v.validate_csv_data(phase_data, sample_rate=5000, filename='ok.csv')
    # не должно быть CRITICAL
    severities = {r.severity for r in results}
    assert ValidationSeverity.CRITICAL not in severities


def test_validate_bad_length():
    v = DataValidator()
    phase_data = {'R':[1.0,2.0], 'S':[], 'T':[]}
    results = v.validate_csv_data(phase_data, sample_rate=5000, filename='short.csv')
    assert any(r.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL) for r in results)
