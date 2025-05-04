from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, Response
from io import BytesIO
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from src.model import ConvolutionalAutoEncoder
import os
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import uvicorn

# Define Prometheus metrics
REQUESTS = Counter('denoise_requests_total', 'Total number of denoise requests')
LATENCY = Histogram('denoise_latency_seconds', 'Latency of denoise requests in seconds')

# Initialize FastAPI app for the denoising model
app = FastAPI(title="Denoising Autoencoder API")

# Prometheus metrics endpoint
@app.get("/metrics")
async def get_metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Load model at startup
model = ConvolutionalAutoEncoder()
model.load_state_dict(torch.load("models/denoising_model.pth", map_location=torch.device('cpu')))
model.eval()

# Preprocessing transform
transform = transforms.Compose([
    transforms.ToTensor()
])

# Postprocessing (Tensor to PIL image)
def tensor_to_pil(tensor):
    tensor = tensor.squeeze(0).detach().clamp(0, 1)
    image = transforms.ToPILImage()(tensor)
    return image

@app.post("/denoise")
async def denoise_image(file: UploadFile = File(...)):
    """ Endpoint to denoise images """
    try:
        # Increment request counter
        REQUESTS.inc()

        # Record the start time for latency calculation
        with LATENCY.time():
            image = Image.open(BytesIO(await file.read())).convert("RGB")
            input_tensor = transform(image).unsqueeze(0)  # shape: [1, 3, 32, 32]

            with torch.no_grad():
                output_tensor = model(input_tensor)

            output_image = tensor_to_pil(output_tensor)

            # Return image as downloadable file
            buffer = BytesIO()
            output_image.save(buffer, format="PNG")
            buffer.seek(0)
            
            original_name = os.path.splitext(file.filename)[0]
            download_name = f"{original_name}_denoise.png"

            headers = {
                "Content-Disposition": f'attachment; filename="{download_name}"'
            }

            return StreamingResponse(buffer, media_type="image/png", headers=headers)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

# Run the combined app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)







