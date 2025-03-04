from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import io
import time
import logging
from contextlib import asynccontextmanager
import os
from pydantic import ValidationError, BaseModel
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
import torch

from app.models import GenerationRequest, GenerationResponse, HealthResponse, ModelInfo, StoryGenerationRequest, StoryGenerationResponse, ComicGenerationRequest, ComicGenerationResponse, HealthCheckResponse
from app.services.diffusion import DiffusionService
from app.services.story import StoryService

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
    
    # Initialize services
    try:
        # Initialize story service
        api_key = os.getenv('OPENAI_API_KEY')
        logging.info(f"API Key: {api_key}")
        try:
            story_service = StoryService(api_key)
            app.state.story_service = story_service
            logging.info("Story service initialized successfully!")
        except Exception as e:
            logging.warning(f"Story service initialization failed: {str(e)}")
            logging.warning("Story generation features will be disabled")

        # Initialize diffusion service with the model ID instead of the full path
        diffusion_service = DiffusionService(
            model_id="flux",  # Use the model ID from our config
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        app.state.diffusion_service = diffusion_service
        
    except Exception as e:
        logging.error(f"Failed to load model: {str(e)}")
        raise e  # Still raise for diffusion service as it's required

    yield

    # Cleanup
    if hasattr(app.state, "diffusion_service"):
        del app.state.diffusion_service

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


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Check if the service is healthy and model is loaded"""
    if not hasattr(app.state, "diffusion_service"):
        return HealthCheckResponse(
            status="unhealthy",
            message="Model not loaded",
            model_info={}
        )
    
    model_info = app.state.diffusion_service.get_model_info()
    return HealthCheckResponse(
        status="healthy",
        message="Service is running normally",
        model_info=model_info
    )


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


@app.post("/generate/comic", response_model=ComicGenerationResponse)
async def generate_comic(request: ComicGenerationRequest):
    try:
        # Check if story service is available
        if not hasattr(app.state, "story_service"):
            # Create Panel objects directly
            panels = [Panel(prompt=f"Panel {i+1}: {request.title}") for i in range(request.num_panels)]
        else:
            # Generate story using story service
            story = await app.state.story_service.generate_story(
                title=request.title,
                num_panels=request.num_panels
            )
            panels = story.panels

        # Generate images for each panel
        generated_panels = []
        for panel in panels:
            try:
                image = app.state.diffusion_service.generate(
                    prompt=panel.prompt,  # Access prompt as attribute
                    model_id=request.model_id,
                    negative_prompt=panel.negative_prompt  # Access negative_prompt as attribute
                )
                
                # Convert image to base64
                image_data = app.state.diffusion_service.image_to_base64(image)
                
                generated_panels.append({
                    "prompt": panel.prompt,
                    "image": image_data
                })
                
            except Exception as e:
                logging.error(f"Error generating panel: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error generating panel: {str(e)}")

        return ComicGenerationResponse(
            title=request.title,
            panels=generated_panels,
            model_id=request.model_id
        )

    except Exception as e:
        logging.error(f"Error generating comic: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
