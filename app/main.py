import os
import shutil
import tempfile
from fastapi import FastAPI, File, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
from app.yolo_process import process_yolo_to_svg
from ultralytics import YOLO
import urllib.request
import pathlib

MODEL_URL = os.getenv("MODEL_URL")  # put a signed/public URL here

def ensure_model_on_disk(path: str):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        return
    if not MODEL_URL:
        raise RuntimeError("MODEL_URL not set and model file missing.")
    print(f"Downloading model from {MODEL_URL} to {p} ...")
    urllib.request.urlretrieve(MODEL_URL, p)  # simple, blocking


app = FastAPI(title="YOLOv11 Segmentation API")

allow_origins = [o.strip() for o in os.getenv("ALLOW_ORIGINS", "http://localhost:3000").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_PATH", "/data/model/yolov11_instance_trained.pt")
model: YOLO | None = None

@app.on_event("startup")
def _load_model():
    global model
    ensure_model_on_disk(MODEL_PATH)   # <--- add this
    model = YOLO(MODEL_PATH)
    # model.to("cpu")


@app.get("/health")
def health():
    return {"ok": True}

@app.post("/segment-svg")
async def segment_svg(file: UploadFile = File(...)):
    # Save to a temp file to hand to OpenCV/YOLO
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
