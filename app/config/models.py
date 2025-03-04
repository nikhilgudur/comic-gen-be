from typing import Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the base path for local models from environment variable or use a default
# Update this path to match your actual LoRA file location
LOCAL_MODELS_DIR = "/app/comic_gen_server/app/models"

AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
    "flux": {
        "model_id": "black-forest-labs/FLUX.1-dev",
        "lora_path": "shreenithi20/flux_lora_sketch_v8",
        "description": "FLUX model with sketch LoRA"
    },
    "flux-base": {
        "model_id": "black-forest-labs/FLUX.1-dev",
        "description": "Base FLUX model without LoRA"
    },
    "sd1.5": {
        "model_id": "runwayml/stable-diffusion-v1-5",
        "description": "Stable Diffusion 1.5"
    },
    "sd-flintstones": {
        "model_id": "runwayml/stable-diffusion-v1-5",
        "lora_path": "ComicGenAI/sd-finetuned-flintstones",
        "description": "Stable Diffusion with Flintstones LoRA"
    },
    "flux-local-lora": {
        "model_id": "black-forest-labs/FLUX.1-dev",
        "local_lora": os.path.join(LOCAL_MODELS_DIR, "flux_lora.safetensors"),
        "description": "FLUX model with local LoRA weights"
    }
}

def get_model_config(model_id: str) -> Dict[str, Any]:
    """Get model configuration by ID"""
    if model_id not in AVAILABLE_MODELS:
        raise ValueError(f"Model ID '{model_id}' not found. Available models: {list(AVAILABLE_MODELS.keys())}")
    
    config = AVAILABLE_MODELS[model_id].copy()
    
    # Verify local LoRA file exists if specified
    if 'local_lora' in config and not os.path.exists(config['local_lora']):
        raise ValueError(f"Local LoRA file not found: {config['local_lora']}")
    
    return config