from pydantic import BaseModel
from typing import List


class Passenger(BaseModel):
    Pclass: int
    Sex: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: int


class PredictionRequest(BaseModel):
    passengers: List[Passenger]
