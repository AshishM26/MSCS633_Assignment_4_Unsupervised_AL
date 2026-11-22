"""Fast tests using synthetic data rather than the private benchmark CSV."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud_detection.config import EXPECTED_COLUMNS
from fraud_detection.data import split_data, validate_data
from fraud_detection.evaluation import evaluate
from fraud_detection.model import build_model, select_device
from fraud_detection.preprocessing import scale_features
from run_experiment import output_directory


@pytest.fixture
def frame():
    rng = np.random.default_rng(42)
    data = pd.DataFrame(rng.normal(size=(100, 31)), columns=EXPECTED_COLUMNS)
    data['Class'] = [0, 1] * 50
    return data


def test_valid_schema(frame):
    validate_data(frame, benchmark=False)


def test_missing_column(frame):
    with pytest.raises(ValueError, match='Expected columns'):
        validate_data(frame.drop(columns='V1'), benchmark=False)


def test_invalid_class(frame):
    frame.loc[0, 'Class'] = 2
    with pytest.raises(ValueError, match='Class'):
        validate_data(frame, benchmark=False)


def test_nonfinite_rejected(frame):
    frame.loc[0, 'V2'] = np.inf
    with pytest.raises(ValueError, match='infinite'):
        validate_data(frame, benchmark=False)


def test_benchmark_counts_rejected(frame):
    with pytest.raises(ValueError, match='benchmark counts'):
        validate_data(frame)


def test_features_and_deterministic_split(frame):
    a, b, y, z = split_data(frame)
    a2, b2, y2, z2 = split_data(frame)
    assert 'Class' not in a and a.shape == (80, 30)
    assert a.equals(a2) and b.equals(b2) and y.equals(y2) and z.equals(z2)


def test_scaling_shape_finite_and_train_only(frame):
    train, test, _, _ = split_data(frame)
    a, b = scale_features(train, test)
    assert a.shape == train.shape and b.shape == test.shape
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert np.allclose(a.mean(axis=0), 0, atol=1e-10)


def test_metrics_from_continuous_scores():
    y = [0, 0, 1, 1]
    scores = [0.1, 0.8, 0.4, 0.9]
    result = evaluate(y, scores, [0, 1, 0, 1])
    assert (result['tn'], result['fp'], result['fn'], result['tp']) == (1, 1, 1, 1)
    assert result['precision'] == result['recall'] == result['f1'] == 0.5
    assert result['roc_auc'] == roc_auc_score(y, scores)
    assert result['average_precision'] == average_precision_score(y, scores)
    assert result['predicted_anomalies'] == 2


def test_device_selection():
    assert select_device('auto') in {'mps', 'cpu'}
    assert select_device('cpu') == 'cpu'
    with pytest.raises(ValueError):
        select_device('bad')


def test_model_configuration_and_tiny_training():
    model = build_model('cpu', epochs=1, batch_size=16)
    assert model.preprocessing is False
    x = np.random.default_rng(2).normal(size=(64, 30))
    model.fit(x)
    scores = model.decision_function(x[:10])
    assert scores.shape == (10,) and np.isfinite(scores).all()


def test_quick_output_separated():
    assert output_directory(True).name == 'quick'
    assert output_directory(True) != output_directory(False)
    assert output_directory(True).is_dir()
