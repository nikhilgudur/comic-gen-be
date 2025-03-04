import requests
import json
import time

print("Testing local LoRA model...")
try:
    data = {
        "title": "A story about a hero cat",
        "num_panels": 1,
        "model_id": "flux-local-lora"
    }
    
    print(f"Sending request with data: {json.dumps(data)}")
    print("This might take a while (up to 5 minutes for first generation)...")
    
    response = requests.post(
        "http://localhost:8001/generate/comic",
        json=data,
        timeout=300  # 5 minutes timeout
    )
    
    print(f"Response status: {response.status_code}")
    if response.status_code == 200:
        print("Success! The local LoRA model works!")
        # Save partial response to avoid printing the entire base64 image
        response_json = response.json()
        if "panels" in response_json and len(response_json["panels"]) > 0:
            # Replace base64 image data with "[IMAGE DATA]" to keep output readable
            for panel in response_json["panels"]:
                if "image_b64" in panel:
                    panel["image_b64"] = "[IMAGE DATA]"
        print(f"Response content (truncated): {json.dumps(response_json, indent=2)}")
    else:
        print(f"Error content: {response.text}")
except Exception as e:
    print(f"Error: {str(e)}") 