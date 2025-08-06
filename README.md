# Insider ML Engineer Challenge - Titanic Dataset

In this challenge, we developed a full API to use it with AI models.

The Titanic Dataset can be found at [Kaggle](https://www.kaggle.com/competitions/titanic/data), but in this case, it is already downloaded in the [`dataset directory`](dataset), due to its small size.

## Project structure

```bash
.
├── Dockerfile
├── README.md
├── compose.yaml
├── dataset
│   ├── api_examples.json (we can test our API using this)
│   ├── gender_submission.csv
│   ├── test.csv
│   └── train.csv
├── docs (Kaggle submissions)
│   ├── rf_submission.csv
│   ├── submissions.png
│   └── svc_submission.csv
├── models
│   ├── best_rf_pipeline.pkl
│   └── best_svc_pipeline.pkl
├── notebooks
│   └── eda.ipynb
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── api
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── core
│   │   ├── __init__.py
│   │   └── model_manager.py
│   ├── main.py
│   ├── train.py
│   └── utils.py
└── tests
    └── test_api.py
```

### API Objective

```mermaid
flowchart TD
    %% Styling
    classDef client fill:#90CAF9,stroke:#1565C0,color:#000
    classDef api fill:#81C784,stroke:#2E7D32,color:#000
    classDef core fill:#FFB74D,stroke:#EF6C00,color:#000
    classDef storage fill:#CE93D8,stroke:#7B1FA2,color:#000
    
    subgraph Client["External Clients"]
        C1[Web Applications]:::client
        C2[API Consumers]:::client
    end
    
    subgraph API["API Layer"]
        direction TB
        AP[FastAPI Application]:::api
        PR[Prediction Routes]:::api
        LR[Load Model Routes]:::api
        HR[History Routes]:::api
        HS[Health Check]:::api
        
        AP --> PR & LR & HR & HS
    end
    
    subgraph Core["Core Services"]
        direction TB
        MM[Model Manager]:::core
        PH[Prediction Handler]:::core
        MH[Model History]:::core
        LH[Logging Service]:::core
        
        MM --> PH & MH
        PH & MH --> LH
    end
    
    subgraph Storage["Storage Layer"]
        direction TB
        MP[Models Directory]:::storage
        HD[(Prediction History)]:::storage
        LD[(Logs)]:::storage
    end
    
    %% Connections
    C1 & C2 --> AP
    PR & LR & HR & HS --> MM
    PH --> MP
    MH --> HD
    LH --> LD
```

## Requirements

- Python 3.12.x

### Preparing the environment

**PS: The following commands may be used on a Linux terminal.**

First, we need to create and activate our virtual environment (venv):

```bash
python3 -m venv venv && source venv/bin/activate
```

Then, using pip, we must download the dependecies from [`requirements.txt`](requirements.txt)

```bash
pip install -r requirements.txt
```

### Git LFS

It is also good to install the Git LFS for uploading models to GitHub.
In Linux systems, especifically Debian based like Ubuntu, we can do this in the terminal running:

```bash
sudo apt install git-lfs
```

Then, we must initialize Git LFS in our project directory:

```bash
git lfs install
```

Now, we need to pull our pickle files:

```bash
git lfs pull
```

This will download the .pkl models in place of the pointer files.

More details about the model, like its evaluation and score can be found in [`this notebook`](notebooks/eda.ipynb).

## Training

From the root project directory, you can run the training using:

```bash
python3 -m src.train
```

The training should take a few minutes. On a laptop with 8 GB RAM, i5-1135G7 processor and Intel Iris Xe Graphics, it took $\approx$ 5 minutes.

## Running the API

Since we are using FastAPI, the API can be tested using

```bash
uvicorn src.main:app --reload
```

### Using Docker

The Dockerfile was created using ```docker init```.  
We can build and run our API Docker image using:

```bash
docker compose up --build
```

Our application will be available at <http://localhost:8000>.

## Testing the API

The default route of the API will be redirected to the docs with Swagger. We can test the API routes right from there.

### /health

Checks the API health.

### /predict

Below we have an example of a passenger input expected for the prediction.  
Note:  

- Some values can be null, we can check the docs accordingly
- `survived` may not be in input:

```json
{
    "PassengerId":661,
    "Pclass":1,
    "Name":"Frauenthal, Dr. Henry William",
    "Sex":"male",
    "Age":50.0,
    "SibSp":2,
    "Parch":0,
    "Ticket":"PC 17611",
    "Fare":133.65,
    "Cabin":null,
    "Embarked":"S"
}
```

The output will be an 0 (not survived) or 1 (survived).

### /load

Expects an input string of the .pkl name file to load a model.

### /history

Displays the history of past inputs and predictions.

## Cloud deploy (not tested)

We can build our image specifying the platform according to our CPU architecture in deploy:  

```bash
docker build --platform=linux/amd64 -t titanic-api .
```

Then, we can push it to our registry:

```bash
docker push https://www.titanic-api.com/api
```
