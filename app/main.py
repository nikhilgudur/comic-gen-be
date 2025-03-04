from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import io
import time
import logging
from contextlib import asynccontextmanager
import os
from pydantic import ValidationError
from typing import Optional
from dotenv import load_dotenv

from app.models import GenerationRequest, GenerationResponse, HealthResponse, ModelInfo, StoryGenerationRequest, StoryGenerationResponse
from app.services.diffusion import DiffusionService
from app.services.story import StoryService
from app.utils.comic import create_comic_strip, comic_strip_to_base64

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Store the diffusion service globally
diffusion_service = None

# Initialize StoryService
story_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global diffusion_service, story_service
    
    # Initialize story service
    try:
        story_service = StoryService()
        logger.info("Story service initialized successfully!")
    except Exception as e:
        logger.warning(f"Story service initialization failed: {str(e)}")
        logger.warning("Story generation features will be disabled")
        story_service = None

    # Initialize diffusion service
    logger.info("Loading Stable Diffusion model...")
    try:
        # Initialize with just the base model, no LoRA by default
        diffusion_service = DiffusionService(
            base_model="runwayml/stable-diffusion-v1-5",
            lora_model=None,  # No LoRA by default
            device=os.getenv("DEVICE", "cuda")
        )
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        diffusion_service = None
        raise e  # Still raise for diffusion service as it's required

    yield

    # Clean up resources on shutdown
    if diffusion_service:
        logger.info("Unloading model...")
        # diffusion_service.unload() - Implement if needed

app = FastAPI(
    title="Stable Diffusion API",
    description="API for generating images using Stable Diffusion with LoRA weights",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_diffusion_service():
    if diffusion_service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return diffusion_service


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the service is healthy and the model is loaded"""
    is_loaded = diffusion_service is not None
    status = "ok" if is_loaded else "error"
    message = "Service is running and model is loaded" if is_loaded else f"Model not loaded. Check server logs for details."
    
    if not is_loaded:
        logger.warning("Health check failed: Model is not loaded")
        
    return {
        "status": status,
        "message": message,
        "timestamp": time.time()
    }


@app.get("/model", response_model=ModelInfo)
async def get_model_info(service: DiffusionService = Depends(get_diffusion_service)):
    """Get information about the loaded model"""
    return service.get_model_info()


@app.post("/generate", response_model=GenerationResponse)
async def generate_image(
    request: GenerationRequest,
    background_tasks: BackgroundTasks,
    service: DiffusionService = Depends(get_diffusion_service)
):
    """Generate an image based on the prompt"""
    try:
        logger.info(f"Generating image with prompt: {request.prompt}")
        start_time = time.time()

        # Generate the image
        image = service.generate(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            width=request.width,
            height=request.height,
            seed=request.seed
        )

        # Convert image to base64
        image_base64 = service.image_to_base64(image)

        generation_time = time.time() - start_time
        logger.info(f"Image generated in {generation_time:.2f} seconds")

        return {
            "success": True,
            "image": image_base64,
            "generation_time": generation_time
        }

    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error generating image: {str(e)}")


@app.post("/generate/download")
async def generate_image_download(
    request: GenerationRequest,
    service: DiffusionService = Depends(get_diffusion_service)
):
    """Generate an image and return it as a downloadable file"""
    try:
        logger.info(
            f"Generating downloadable image with prompt: {request.prompt}")
        start_time = time.time()

        # Generate the image
        image = service.generate(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            width=request.width,
            height=request.height,
            seed=request.seed
        )

        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        generation_time = time.time() - start_time
        logger.info(
            f"Downloadable image generated in {generation_time:.2f} seconds")

        # Return image as a downloadable file
        filename = f"generated_{int(time.time())}.png"
        return StreamingResponse(
            content=img_byte_arr,
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except Exception as e:
        logger.error(f"Error generating downloadable image: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error generating image: {str(e)}")


@app.post("/generate/comic", response_model=StoryGenerationResponse)
async def generate_comic(request: StoryGenerationRequest):
    """Generate a complete comic with story and images"""
    if story_service is None:
        raise HTTPException(
            status_code=503, 
            detail="Story service not available. Please set OPENAI_API_KEY environment variable to enable story generation."
        )
    
    try:
        # Generate the story
        story = story_service.generate_story(request.title, request.num_panels)
        
        # Create image prompts for each panel using Flintstones style
        panels = story_service.create_image_prompts(story, style="prehistoric Flintstones style")
        
        # Generate images for each panel
        generated_images = []
        base_seed = 42  # Using the same base seed as in POC for consistency
        
        for panel in panels:
            # Calculate seed based on panel number for consistency
            try:
                panel_num = int(panel.panel_num)
                seed = base_seed + panel_num
            except ValueError:
                # Fallback if panel_num isn't a valid integer
                seed = base_seed
            
            logger.info(f"Generating Panel {panel.panel_num} with seed {seed}...")
            
            # Generate the image with specified seed
            image = diffusion_service.generate(
                prompt=panel.image_prompt,
                negative_prompt="no speech bubbles, no dialogues, no text, no captions, no quotes",
                num_inference_steps=50,
                guidance_scale=7.5,
                seed=seed
            )
            generated_images.append(image)
            # Convert image to base64 and add to panel
            panel.image = diffusion_service.image_to_base64(image)
        
        # Create comic strip
        captions = [panel.title for panel in panels]
        comic_strip = create_comic_strip(generated_images, captions)
        comic_strip_base64 = comic_strip_to_base64(comic_strip)
        
        return {
            "success": True,
            "story": story,
            "panels": panels,
            "comic_strip": comic_strip_base64
        }
        
    except Exception as e:
        logger.error(f"Error generating comic: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating comic: {str(e)}"
        )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
