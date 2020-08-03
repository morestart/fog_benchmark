import os


class Config:
    path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

    x_scaler_path = os.path.join(path, 'model/x_scaler.pkl')
    y_scaler_path = os.path.join(path, 'model/y_scaler.pkl')

    predict_model_path = os.path.join(path, 'model/mse-青岛全 预报2H-3-4-25-257(4e).h5')
    data_path = os.path.join(path, 'data/xxx.csv')

    data_index = 'time'

    step = 3
