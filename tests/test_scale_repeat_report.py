import json
import numpy as np
from scripts.summarize_scale_agreement_repeat import json_ready
from src.scale_repeat_statistics import paired_interval


def test_numpy_scalar_report_exports_without_numeric_changes():
    value = {'adequate': np.bool_(True), 'small': np.bool_(False), 'count': np.int64(5),
        'value': np.float64(.125), 'array': np.array([1., 2.])}
    restored = json.loads(json.dumps(json_ready(value), allow_nan=False))
    assert restored == {'adequate': True, 'small': False, 'count': 5, 'value': .125, 'array': [1., 2.]}


def test_real_interval_report_is_json_serializable():
    result = paired_interval(np.linspace(-1, 1, 30), np.arange(30), 200, 42)
    clean = json_ready(result)
    assert json.loads(json.dumps(clean, allow_nan=False)) == clean
    assert clean['delta_r1_pp'] == result['delta_r1_pp']
    assert clean['percentile_95_ci_pp'] == result['percentile_95_ci_pp']
