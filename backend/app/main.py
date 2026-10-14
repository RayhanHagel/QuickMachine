from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.routers.data import router as data_router
from backend.app.routers.config import router as config_router
from backend.app.routers.training import router as training_router

app = FastAPI(
    title="QuickMachine API",
    description="Automated Machine Learning & Model Stacking API backend",
    version="2.0.0"
)

# Enable CORS for Next.js frontend (development on localhost:3000, 3001, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data_router)
app.include_router(config_router)
app.include_router(training_router)


@app.get("/")
def root():
    return {
        "app": "QuickMachine API",
        "version": "2.0.0",
        "docs": "/docs",
        "status": "online"
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}
