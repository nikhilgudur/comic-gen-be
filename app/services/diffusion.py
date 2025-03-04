from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler
from diffusers.utils import load_image
import torch
from typing import Optional, Dict, Any, List, Union
import base64
import io
from PIL import Image
import logging
import random
import time
import os
from pathlib import Path
from dotenv import load_dotenv
from ..config.models import get_model_config, AVAILABLE_MODELS

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class DiffusionService:
    def __init__(self, model_id="flux", device=None):
        self.device = device or os.getenv('DEVICE', "cuda")
        self.models_cache = {}  # Cache to store loaded models
        
        # Initialize with default model
        self._load_default_model(model_id)

    def _load_default_model(self, model_id: str):
        """Load the default model by ID"""
        try:
            model_config = get_model_config(model_id)
            logging.info(f"Loading default model: {model_config['model_id']}")
            self.current_model_id = model_id
            self.current_pipe = self._load_model(model_config)
        except Exception as e:
            logging.error(f"Failed to load model {model_id}: {str(e)}")
            raise

    def _load_model(self, model_config: Dict[str, Any]) -> Any:
        """Load a model based on configuration"""
        model_id = model_config['model_id']
        lora_path = model_config.get('lora_path')
        local_lora = model_config.get('local_lora')
        
        # Create a unique cache key
        cache_key = f"{model_id}_{lora_path}_{local_lora}"
        
        if cache_key in self.models_cache:
            logging.info(f"Using cached model: {model_id}")
            return self.models_cache[cache_key]

        logging.info(f"Loading model: {model_id}")
        try:
            pipe = AutoPipelineForText2Image.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                use_safetensors=True
            )

            # Load LoRA weights if specified
            if local_lora:
                logging.info(f"Loading local LoRA from: {local_lora}")
                pipe.load_lora_weights(local_lora)
            elif lora_path:
                logging.info(f"Loading LoRA from Hub: {lora_path}")
                pipe.load_lora_weights(lora_path)

            if self.device == "cuda":
                pipe = pipe.to(self.device)
            pipe.enable_attention_slicing()
            
            self.models_cache[cache_key] = pipe
            return pipe
            
        except Exception as e:
            logging.error(f"Error loading model {model_id}: {str(e)}")
            raise

    def generate(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None
    ) -> Image.Image:
        """Generate an image based on the prompt and model ID"""
        try:
            # Load specific model if requested
            if model_id and model_id != self.current_model_id:
                model_config = get_model_config(model_id)
                pipe = self._load_model(model_config)
            else:
                pipe = self.current_pipe

            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            else:
                generator = torch.Generator(device=self.device).manual_seed(
                    random.randint(0, 2147483647))

            # Generate the image
            with torch.no_grad():
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    generator=generator
                )

            return result.images[0]
            
        except Exception as e:
            logging.error(f"Error in generate: {str(e)}")
            raise

    def image_to_base64(self, image: Image.Image, format: str = "PNG") -> str:
        """Convert a PIL Image to a base64 string"""
        buffered = io.BytesIO()
        image.save(buffered, format=format)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/{format.lower()};base64,{img_str}"

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models"""
        return {
            "current_model": self.current_model_id,
            "available_models": AVAILABLE_MODELS,
            "device": self.device,
            "supported_features": [
                "txt2img",
                "negative_prompt",
                "custom_seeds"
            ],
            "max_image_size": {"width": 1024, "height": 1024}
        }
