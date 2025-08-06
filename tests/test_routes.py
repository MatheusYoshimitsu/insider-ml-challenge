from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_prediction_missing_field():
    data = {
        "Pclass": 1,
        "Name": "John Doe",
        "Sex": "male",
        # "Age" missing on purpose
        "SibSp": 0,
        "Parch": 0,
        "Ticket": "12345",
        "Fare": 15.5,
        "Cabin": None,
        "Embarked": "C",
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200 or response.status_code == 400


def test_root_redirect():
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"
