import pytest
from src.api.schemas import PassengerData


def test_valid_passenger_data():
    data = {
        "Pclass": 2,
        "Name": "Doe, Mr. John",
        "Sex": "male",
        "Age": 30,
        "SibSp": 0,
        "Parch": 0,
        "Ticket": "A/5 21171",
        "Fare": 10.5,
        "Cabin": None,
        "Embarked": "S",
    }
    passenger = PassengerData(**data)
    assert passenger.Pclass == 2


def test_invalid_extra_field():
    with pytest.raises(Exception):
        PassengerData(
            Pclass=1,
            Name="John Doe",
            Sex="male",
            Age=22,
            SibSp=0,
            Parch=0,
            Ticket="12345",
            Fare=7.5,
            Cabin=None,
            Embarked="C",
            ExtraField="not allowed",
        )
