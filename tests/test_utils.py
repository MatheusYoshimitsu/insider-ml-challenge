import pandas as pd
from src.utils import feature_engineering


def test_feature_engineering_basic_transformations():
    df = pd.DataFrame(
        [
            {
                "PassengerId": 1,
                "Name": "Smith, Mr. John",
                "Pclass": 1,
                "Sex": "male",
                "Age": None,
                "SibSp": 0,
                "Parch": 0,
                "Ticket": "12345",
                "Fare": None,
                "Cabin": None,
                "Embarked": None,
            },
            {
                "PassengerId": 2,
                "Name": "Doe, Mrs. Jane",
                "Pclass": 2,
                "Sex": "female",
                "Age": 28,
                "SibSp": 1,
                "Parch": 0,
                "Ticket": "67890",
                "Fare": 100,
                "Cabin": "C85",
                "Embarked": "S",
            },
        ]
    )

    result = feature_engineering(df)

    assert "norm_fare" in result.columns
    assert "cabin_multiple" in result.columns
    assert "name_title" in result.columns
    assert "Pclass" in result.columns
    assert result["Age"].notnull().all()
