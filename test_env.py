import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print(f"LOCAL_MODELS_DIR: {os.getenv('LOCAL_MODELS_DIR', 'Not set')}")
print(f"SD_BASE_MODEL: {os.getenv('SD_BASE_MODEL', 'Not set')}")
print(f"SD_LORA_MODEL: {os.getenv('SD_LORA_MODEL', 'Not set')}") 