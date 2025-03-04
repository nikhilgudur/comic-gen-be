from diffusers import DiffusionPipeline, StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import load_image
import torch
from typing import Optional, Dict, Any, List
import base64
import io
from PIL import Image
import logging
import random
import time
import os

logger = logging.getLogger(__name__)


class DiffusionService:
    def __init__(
        self,
        base_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
        lora_model: Optional[str] = "ComicGenAI/sd-finetuned-flintstones",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16  # Use float32 for CPU
    ):
        self.base_model = base_model
        self.lora_model = lora_model
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.dtype = dtype if self.device == "cuda" else torch.float32

        # Load the model
        logger.info(f"Loading base model: {base_model}")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            base_model,
            torch_dtype=self.dtype,
            safety_checker=None,  # Disable safety checker for speed
            requires_safety_checker=False
        )

        # Use the DPMSolver++ scheduler for faster inference
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True
        )

        # Load LoRA weights if provided
        if lora_model:
            logger.info(f"Loading LoRA weights: {lora_model}")
            self.pipe.load_lora_weights(lora_model)

        # Move to device
        self.pipe = self.pipe.to(self.device)

        # Enable memory efficient attention if using CUDA
        if self.device == "cuda":
            self.pipe.enable_xformers_memory_efficient_attention()

        # Optional: enable attention slicing for lower memory usage
        self.pipe.enable_attention_slicing()

        logger.info(f"Model loaded on {self.device} with dtype {self.dtype}")

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None
    ) -> Image.Image:
        """Generate an image based on the prompt"""
        # Set the seed for reproducibility
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = torch.Generator(device=self.device).manual_seed(
                random.randint(0, 2147483647))

        # Generate the image
        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator
            )

        return result.images[0]

    def image_to_base64(self, image: Image.Image, format: str = "PNG") -> str:
        """Convert a PIL Image to a base64 string"""
        buffered = io.BytesIO()
        image.save(buffered, format=format)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/{format.lower()};base64,{img_str}"

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "base_model": self.base_model,
            "lora_model": self.lora_model,
            "device": self.device,
            "supported_features": ["txt2img", "negative_prompt", "custom_seeds"],
            "max_image_size": {"width": 1024, "height": 1024}
        }
