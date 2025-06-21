import fal_client
from redis import Redis
import logging
import os
import json
from dotenv import load_dotenv

load_dotenv()
os.environ["FAL_KEY"] = "671b3239-6a7c-4449-a85e-caa161fe3351:631d5730d85692de182d398b1b072496"

redis_conn = Redis(host="redis", port=6379)
logging.basicConfig(level=logging.INFO)

RESULT_TTL = 3600

async def generate_image(
    prompt: str,
    client_id: str,
    *,
    image_size: str = "landscape_4_3",
    width: int = None,
    height: int = None,
    num_images: int = 1,
    seed: int = None,
    sync_mode: bool = False,
    enable_safety_checker: bool = True,
    safety_tolerance: str = "2",
    output_format: str = "jpeg",
):
    """
    Generate images using fal-client with configurable parameters.

    Arguments:
      model_name (str): model identifier, e.g. 'fal-ai/flux-pro/v1.1'
      prompt (str): the text prompt
      client_id (str): id to store Redis result
    Keyword-only arguments:
      image_size (str): one of 'square_hd', 'square', 'portrait_4_3', 'portrait_16_9', 'landscape_4_3', 'landscape_16_9'
      width, height (int): for custom size instead of enum
      num_images (int): number of images
      seed (int): for reproducibility
      sync_mode (bool): sync generation mode
      enable_safety_checker (bool), safety_tolerance (str): NSFW safety settings
      output_format (str): 'jpeg' or 'png'
    """
    try:
        logging.info(f"Generating image for {client_id} with model ...")

        args = {
            "prompt": prompt,
            "num_images": num_images,
            "enable_safety_checker": enable_safety_checker,
            "safety_tolerance": safety_tolerance,
            "output_format": output_format
        }

        if width and height:
            args["image_size"] = {"width": width, "height": height}
        else:
            args["image_size"] = image_size

        if seed is not None:
            args["seed"] = seed

        if sync_mode:
            args["sync_mode"] = True

        handler = fal_client.submit("fal-ai/flux-pro/v1.1", arguments=args)
        result = handler.get()

        redis_conn.setex(f"result:{client_id}", RESULT_TTL, json.dumps(result))
        logging.info(f"Image generation completed for client {client_id}")
    except Exception as e:
        msg = str(e)
        redis_conn.setex(f"result:{client_id}", RESULT_TTL, msg)
        logging.error(f"Failed for {client_id}: {msg}")
