import pickle
from pathlib import Path

# project_root = 2 levels up from this file (report/ → project root)
project_root = Path(__file__).resolve().parent.parent

# model_path points to assets/model.pkl
model_path = project_root / "assets" / "model.pkl"


def load_model():
    with model_path.open('rb') as file:
        model = pickle.load(file)
    return model
