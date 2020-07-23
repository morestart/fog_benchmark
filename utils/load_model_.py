import joblib
from tensorflow.keras.models import load_model

from config.config import Config


def load_scaler_model():
    x_scaler = joblib.load(Config.x_scaler_path)
    y_scaler = joblib.load(Config.y_scaler_path)
    pre_model = load_model(Config.predict_model_path)

    return x_scaler, y_scaler, pre_model
