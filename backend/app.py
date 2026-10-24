"""FastAPI application for LSM sign recognition."""

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from inference import InvalidVideoError, predire


app = FastAPI(title="Traduction LSM", version="1.0.0")
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"


@app.post("/predire")
def predire_endpoint(file: UploadFile = File(...)):
    video_bytes = file.file.read()
    try:
        return predire(video_bytes)
    except InvalidVideoError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


app.mount(
    "/",
    StaticFiles(directory=str(FRONTEND_DIR), html=True, check_dir=False),
    name="frontend",
)
