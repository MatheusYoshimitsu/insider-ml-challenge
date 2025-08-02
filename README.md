# Insider ML Engineer Challenge - Titanic Dataset

In this challenge, we are going to develop a full API to use it with AI.

The Titanic Dataset can be found at [Kaggle](https://www.kaggle.com/competitions/titanic/data), but in this case, it is already downloaded in the [`dataset directory`](dataset), due to its small size.

## Requirements

- Python 3.10+

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

More details about the model, like its evaluation and score can be found in [`this notebook`](notebooks/main.ipynb).

## Running the API

```bash
uvicorn src.main:app --reload
```
