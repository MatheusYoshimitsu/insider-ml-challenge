from pydantic import BaseModel
from pydantic.config import ConfigDict


class PassengerData(BaseModel):
    """
    Schema representing passenger data for prediction.

    Attributes:
        Pclass (int): Passenger class (1, 2, 3).
        Name (str): Full name of the passenger.
        Sex (str): Gender of the passenger ('male', 'female').
        Age (float | None): Age of the passenger. Can be null.
        SibSp (int): Number of siblings/spouses of the passenger.
        Parch (int): Number of parents/children of the passenger.
        Ticket (str): Ticket number.
        Fare (float | None): Fare paid for the ticket. Can be null.
        Cabin (str | None): Cabin number. Can be null.
        Embarked (str | None): Port of embarkation ('C', 'Q', 'S'). Can be null.
    """

    Pclass: int
    Name: str
    Sex: str
    Age: float | None = None
    SibSp: int
    Parch: int
    Ticket: str
    Fare: float | None = None
    Cabin: str | None = None
    Embarked: str | None = None

    model_config = ConfigDict(extra="forbid")
