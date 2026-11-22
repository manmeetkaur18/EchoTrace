from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.media import router as media_router
from app.routers.rag import router as rag_router

app = FastAPI(
    title="EchoTrace",
    description="AI-powered multimedia search and evidence retrieval platform",
    version="1.0.0"
)

# ---------------------------------------------------------
# CORS Configuration for Local Development & Frontend
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
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
