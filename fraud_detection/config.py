"""Configuration for the credit-card benchmark."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "data" / "creditcard.csv"
FEATURES = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
EXPECTED_COLUMNS = [*FEATURES, "Class"]
RANDOM_STATE = 42
TEST_SIZE = 0.2
CONTAMINATION = 492 / 284807
EPOCHS = 20
BATCH_SIZE = 256
