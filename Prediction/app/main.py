from fastapi import FastAPI
from app.handler import router
from app.create_model import create_model

def get_application() -> FastAPI:
    application = FastAPI()
    application.include_router(router)
    create_model()
    return application

app = get_application()