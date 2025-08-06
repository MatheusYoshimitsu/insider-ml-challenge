from fastapi import FastAPI
from src.api.routes import router

app = FastAPI()
"""FastAPI application instance for the Titanic ML API."""

app.include_router(router)
