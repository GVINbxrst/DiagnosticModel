import numpy as np

from src.utils.serialization import dump_float32_array, load_float32_array


def test_round_trip_serialization():
    arr = np.linspace(-1, 1, 128, dtype=np.float32)
    data = dump_float32_array(arr)
    assert isinstance(data, (bytes, bytearray))
    restored = load_float32_array(data)
    assert restored is not None
    assert restored.dtype == np.float32
    np.testing.assert_array_equal(arr, restored)


def test_none_and_empty_cases():
    assert dump_float32_array(None) is None
    empty = np.array([], dtype=np.float32)
    assert dump_float32_array(empty) is None
    assert load_float32_array(None) is None
    assert load_float32_array(b"") is None
