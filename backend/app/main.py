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
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

# Allow local development hosts including loopback and private LAN addresses (e.g. 192.168.x.x)
LOCAL_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+)(:\d+)?$"

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=LOCAL_ORIGIN_REGEX,
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
