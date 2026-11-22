"""Feature scaling learned exclusively from the training partition."""
import numpy as np
from sklearn.preprocessing import StandardScaler


def scale_features(train, test):
    """Fit on train and apply the same transformation to train and test."""
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    if not np.isfinite(train_scaled).all() or not np.isfinite(test_scaled).all():
        raise ValueError("Preprocessing produced non-finite values")
    return train_scaled, test_scaled
