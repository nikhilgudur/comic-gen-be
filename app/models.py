from pydantic import BaseModel, Field, validator, field_validator
from typing import Optional, List, Dict, Any
import random
import time


class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for image generation")
    negative_prompt: Optional[str] = Field(
        default="low quality, bad anatomy, worst quality, low effort",
        description="Text prompt for things to avoid in generation"
    )
    num_inference_steps: Optional[int] = Field(
        default=50, ge=1, le=150,
        description="Number of denoising steps (higher = better quality but slower)"
    )
    guidance_scale: Optional[float] = Field(
        default=7.5, ge=1.0, le=20.0,
        description="How closely the image should follow the prompt"
    )
    width: Optional[int] = Field(
        default=512, ge=256, le=1024, multiple_of=8,
        description="Width of the generated image"
    )
    height: Optional[int] = Field(
        default=512, ge=256, le=1024, multiple_of=8,
        description="Height of the generated image"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for generation (for reproducibility)"
    )

    @field_validator('seed', mode='before')
    @classmethod
    def set_seed(cls, seed):
        if seed is None:
            return random.randint(0, 2147483647)
        return seed


class GenerationResponse(BaseModel):
    success: bool
    image: Optional[str] = None  # Base64 encoded image
    generation_time: Optional[float] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    message: str
    timestamp: float = Field(default_factory=time.time)


class ModelInfo(BaseModel):
    base_model: str
    lora_model: Optional[str] = None
    device: str
    supported_features: List[str] = []
    max_image_size: Dict[str, int] = Field(
        default_factory=lambda: {"width": 1024, "height": 1024})


class StoryGenerationRequest(BaseModel):
    title: str = Field(..., description="Title for the comic story")
    num_panels: int = Field(
        default=5, ge=1, le=10,
        description="Number of panels in the comic"
    )


class PanelInfo(BaseModel):
    panel_num: str
    title: str
    description: str
    image_prompt: str
    image: Optional[str] = None  # Base64 encoded image


class StoryGenerationResponse(BaseModel):
    success: bool
    story: str
    panels: List[PanelInfo]
    error: Optional[str] = None


class Panel(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None


class Story(BaseModel):
    title: str
    panels: List[Panel]


class ModelConfig(BaseModel):
    model_id: Optional[str] = None
    local_weights: Optional[str] = None
    lora_path: Optional[str] = None


class ComicGenerationRequest(BaseModel):
    title: str
    num_panels: int
    model_id: Optional[str] = "flux"  # Default to flux model


class ComicGenerationResponse(BaseModel):
    title: str
    panels: List[Dict[str, str]]
    model_id: str


class HealthCheckResponse(BaseModel):
    status: str
    model_info: Dict[str, Any]
    message: str = "Service is running"  # Add default message
