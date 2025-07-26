import joblib


class TitanicModel:
    def __init__(self):
        self.model = None

    def load_model(self, path: str):
        self.model = joblib.load(path)

    def predict(self, data: list):
        if not self.model:
            raise ValueError("Could not load model")
        return self.model.predict(data).tolist()
