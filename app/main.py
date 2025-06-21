from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from redis import Redis
from rq import Queue, Retry
from . import tasks  # make sure tasks.py has the updated function that accepts all parameters
import uuid
import logging
import asyncio
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI()
logging.basicConfig(level=logging.INFO)

# Redis connection
redis_conn = Redis(host="redis", port=6379, decode_responses=True)
queue = Queue("image_requests", connection=redis_conn)

active_connections = {}
client_result_keys = set()

# Request body model
class ImageRequest(BaseModel):
    prompt: str
    model_name: Optional[str] = "fal-ai/flux-pro/v1.1"
    image_size: Optional[str] = "landscape_4_3"
    num_images: Optional[int] = 1
    output_format: Optional[str] = "jpeg"
    seed: Optional[int] = None
    sync_mode: Optional[bool] = False
    enable_safety_checker: Optional[bool] = True
    safety_tolerance: Optional[str] = "2"
    width: Optional[int] = None
    height: Optional[int] = None

def generate_client_id():
    return str(uuid.uuid4())

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket connection to send real-time results."""
    await websocket.accept()
    active_connections[client_id] = websocket
    logging.info(f"WebSocket connected: {client_id}")

    try:
        while True:
            await websocket.send_text("ping")
            await asyncio.sleep(15)
            await websocket.receive_text()  # Keep connection alive
    except WebSocketDisconnect:
        active_connections.pop(client_id, None)
        logging.info(f"WebSocket disconnected: {client_id}")

@app.post("/generate/")
async def generate_image(request: ImageRequest):
    """Queue an image generation task with parameters."""
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    client_id = generate_client_id()
    client_result_keys.add(client_id)

    # Prepare job arguments
    job_args = {
        "model_name": request.model_name,
        "prompt": request.prompt,
        "client_id": client_id,
        "image_size": request.image_size,
        "num_images": request.num_images,
        "output_format": request.output_format,
        "seed": request.seed,
        "sync_mode": request.sync_mode,
        "enable_safety_checker": request.enable_safety_checker,
        "safety_tolerance": request.safety_tolerance,
        "width": request.width,
        "height": request.height
    }

    # Queue the task
    job = queue.enqueue(tasks.generate_image, **job_args, retry=Retry(max=4))
    logging.info(f"Job queued for client {client_id}")
    return {"job_id": job.id, "client_id": client_id, "message": "Image generation job queued."}

@app.get("/result/{client_id}")
async def get_result(client_id: str):
    """Fetch result if available."""
    result = redis_conn.get(f"result:{client_id}")
    if result:
        return {"status": "done", "result": result}

    return {"status": "pending"}

async def monitor_results():
    """Background task to push completed results via WebSocket."""
    while True:
        await asyncio.sleep(2)
        for client_id in list(client_result_keys):
            result = redis_conn.get(f"result:{client_id}")
            if result:
                websocket = active_connections.get(client_id)
                if websocket:
                    try:
                        await websocket.send_text(f"Result Ready: {result}")
                        logging.info(f"Result pushed to client {client_id}")
                    except Exception as e:
                        logging.error(f"Failed to send result to {client_id}: {e}")
                client_result_keys.remove(client_id)

@app.on_event("startup")
async def startup_event():
    """Start background result monitoring task."""
    asyncio.create_task(monitor_results())
