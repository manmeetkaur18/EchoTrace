from fastapi import FastAPI
from app.api.media import router as media_router
from app.routers.rag import router as rag_router

app = FastAPI(
    title="EchoTrace",
    description="AI-powered multimedia search and evidence retrieval platform",
    version="1.0.0"
)

# Register API routers
app.include_router(media_router)
app.include_router(rag_router)


@app.get("/")
def home():
    return {
        "message": "Welcome to EchoTrace!",
        "status": "Backend is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }
