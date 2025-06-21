from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from redis import Redis
from rq import Queue, Retry
from . import tasks
import uuid
import logging
import asyncio
from pydantic import BaseModel

app = FastAPI()
logging.basicConfig(level=logging.INFO)

class ImageRequest(BaseModel):
  prompt: str
  
  
# Connect to Redis (host "redis" because it runs inside Docker)
redis_conn = Redis(host="redis", port=6379, decode_responses=True)

# Create a queue for handling image requests
queue = Queue("image_requests", connection=redis_conn)

# Store active WebSocket connections
active_connections = {}
client_result_keys = set()

def generate_client_id():
  return str(uuid.uuid4())

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str ):
  """WebSocket connection to send results to users directly."""
  await websocket.accept()
  active_connections[client_id] = websocket
  logging.info(f"webSocket connected: {client_id}")
  
  try:
    while True:
      await websocket.send_text("ping")
      await websockrt.sleep(15)
      await websocket.receive_text()   # Keep connection open
  except WebSocketDisconnect:
    active_connections.pop(client_id, None)
    logging.info(f"webSocket disconnected: {client_id}")



@app.post("/generate/")
async def generate_image(request: ImageRequest):
  """Add an image generation request to the queue"""
  if not request.prompt:
    raise HTTPException(status_code=400, detail="Prompt cannot be empty")

  # Generate a unique client_id if not provided
  client_id = generate_client_id()
  client_result_keys.add(client_id)
    
  # Add job to queue
  job = queue.enqueue(tasks.generate_image, request.prompt, client_id, retry=Retry(max=4))
  logging.info(f"job queued for client {client_id}")
  return {"job_id": job.id, "client_id": client_id, "message": "Image generation job queued." }


@app.get("/result/{client_id}")
async def get_result(client_id: str):
  result = redis_conn.get(f"result: {client_id}")
  if result:
    return {"status": "done",  "result": result}

  return {"status":  "pending"}

async def monitor_results():
  while True:
    await asyncio.sleep(2)
    for client_id in list(client_result_keys):
      result = redis_conn.get(f"result: {client_id}")
      if result:
        websocket = active_connections.get(client_id)
        if websocket:
          try:
            await websocket.send_text(f"Result Ready: {result}")
            logging.info(f"Result pushed to client {client_id}")
          except Exception as e :
            logging.error(f"Failed to send result to {client_id}: {e}")
      client_result_keys.remove(client_id)


@app.on_event("startup")
async def startup_event():
  asyncio.create_task(monitor_results())



    

