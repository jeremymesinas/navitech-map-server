import os
import shutil
import tempfile
import pathlib
import urllib.request

from fastapi import FastAPI, File, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

from app.yolo_process import process_yolo_to_svg

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_SOURCE = os.getenv(
    "MODEL_SOURCE", "app/data/model/yolov11_instance_trained.pt"
)  # local path in repo or public URL
MODEL_PATH = os.getenv(
    "MODEL_PATH", "/opt/render/project/src/app/data/model/yolov11_instance_trained.pt"
)  # final location where model is stored

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def ensure_model_on_disk(dest_path: str):
    """Make sure the YOLO model file exists at dest_path."""
    p = pathlib.Path(dest_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if p.exists():
        print(f"Model already present at {p}")
        return

    if MODEL_SOURCE.startswith(("http://", "https://")):
        print(f"Downloading model from {MODEL_SOURCE} to {p} ...")
        urllib.request.urlretrieve(MODEL_SOURCE, p)
    else:
        print(f"Copying model from {MODEL_SOURCE} to {p} ...")
        shutil.copy(MODEL_SOURCE, p)

# -----------------------------------------------------------------------------
# FastAPI App
# -----------------------------------------------------------------------------
app = FastAPI(title="YOLOv11 Segmentation API")

# Configure CORS
allow_origins = [
    o.strip()
    for o in os.getenv("ALLOW_ORIGINS", "http://localhost:3000").split(",")
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global YOLO model object
model: YOLO | None = None

@app.on_event("startup")
def _load_model():
    global model
    ensure_model_on_disk(MODEL_PATH)
    model = YOLO(MODEL_PATH)
    # model.to("cpu")  # Uncomment if forcing CPU

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/segment-svg")
async def segment_svg(file: UploadFile = File(...)):
    """Accepts an uploaded image and returns YOLO segmentation as an SVG."""
    suffix = os.path.splitext(file.filename)[1] or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        svg = process_yolo_to_svg(tmp_path, model)  # returns an SVG string
        return Response(content=svg, media_type="image/svg+xml")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
