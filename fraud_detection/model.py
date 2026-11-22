"""Construct the PyOD model and select an available compute device."""
import torch
from pyod.models.auto_encoder import AutoEncoder

from .config import BATCH_SIZE, CONTAMINATION, EPOCHS, RANDOM_STATE


def select_device(requested: str) -> str:
    """Choose MPS if available, otherwise CPU; explicit MPS must exist."""
    available = torch.backends.mps.is_available()
    if requested == 'auto':
        return 'mps' if available else 'cpu'
    if requested == 'mps' and not available:
        raise ValueError("MPS was requested but is not available")
    if requested not in {'cpu', 'mps'}:
        raise ValueError(f"Unsupported device: {requested}")
    return requested


def build_model(device: str, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE):
    """Build a reconstruction model; external scaling prevents double scaling."""
    return AutoEncoder(contamination=CONTAMINATION, preprocessing=False, lr=0.001,
                       epoch_num=epochs, batch_size=batch_size, device=device,
                       random_state=RANDOM_STATE, hidden_neuron_list=[64, 32],
                       hidden_activation_name='relu', batch_norm=True,
                       dropout_rate=0.2, verbose=1)
