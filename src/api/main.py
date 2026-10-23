from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from typing import Optional
import shutil
import os
import time
from datetime import datetime

from src.database.mongo import db, ping_db
from src.embeddings.clip_model import clip_embedder
from src.embeddings.similarity import find_top_k_matches
from src.utils.mlflow_logger import log_inference_event

@asynccontextmanager
async def lifespan(app: FastAPI):
    await ping_db()
    os.makedirs("static/uploads", exist_ok=True)
    yield

app = FastAPI(title="SmartFind API", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return {"message": "SmartFind MLOps API is running!"}

@app.post("/reports/found")
async def report_found_item(
    description: str = Form(...),
    image: Optional[UploadFile] = File(None)
):
    start_time = time.time()
    
    # Conditional processing based on whether an image was provided
    if image is not None:
        file_path = f"static/uploads/found_{datetime.now().timestamp()}_{image.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        # Generate multimodal embedding
        embedding = clip_embedder.get_multimodal_embedding(description, file_path)
    else:
        file_path = None
        # Generate text-only embedding
        embedding = clip_embedder.get_text_embedding(description)

    inference_time_ms = round((time.time() - start_time) * 1000, 2)

    found_doc = {
        "description": description,
        "image_path": file_path,
        "embedding": embedding,
        "type": "found",
        "created_at": datetime.utcnow().isoformat()
    }
    result = await db["found_reports"].insert_one(found_doc)

    # Log execution metrics to MLflow
    log_inference_event(
        event_type="found",
        description=description,
        inference_time_ms=inference_time_ms
    )

    return {"status": "success", "id": str(result.inserted_id)}

@app.post("/reports/lost")
async def report_lost_item(
    description: str = Form(...),
    image: Optional[UploadFile] = File(None)
):
    start_time = time.time()
    
    # Conditional processing based on whether an image was provided
    if image is not None:
        file_path = f"static/uploads/lost_{datetime.now().timestamp()}_{image.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        # Generate multimodal embedding
        embedding = clip_embedder.get_multimodal_embedding(description, file_path)
    else:
        file_path = None
        # Generate text-only embedding
        embedding = clip_embedder.get_text_embedding(description)

    inference_time_ms = round((time.time() - start_time) * 1000, 2)

    lost_doc = {
        "description": description,
        "image_path": file_path,
        "embedding": embedding,
        "type": "lost",
        "created_at": datetime.utcnow().isoformat()
    }
    result = await db["lost_reports"].insert_one(lost_doc)

    # Search against existing Found reports
    found_cursor = db["found_reports"].find({})
    found_reports = await found_cursor.to_list(length=1000)
    top_matches = find_top_k_matches(embedding, found_reports, k=5)

    top_score = top_matches[0]["similarity_score"] if top_matches else 0.0

    # Log execution metrics to MLflow
    log_inference_event(
        event_type="lost",
        description=description,
        inference_time_ms=inference_time_ms,
        top_matches_count=len(top_matches),
        top_score=top_score
    )

    return {
        "status": "success",
        "lost_report_id": str(result.inserted_id),
        "top_5_matches": top_matches
    }