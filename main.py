from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input
import numpy as np
from pathlib import Path
import io
from PIL import Image

app = FastAPI()

# Setup templates directory
templates = Jinja2Templates(directory="templates")

# Load the model once at startup
MODEL_PATH = r'model\serious_res_net_50_224x224_95_acc_ai_or_real_keras_file.keras' 
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"preprocess_input": preprocess_input})

print("Model loaded successfully!")


@app.get("/")
async def home(request: Request):
    """Render the main page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Process uploaded image and return AI generation probability
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            return JSONResponse(
                status_code=400,
                content={"error": "File must be an image (JPEG/JPG)"}
            )
        
        # Read image file
        contents = await file.read()
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to target size
        image = image.resize((224, 224))
        
        # Convert to array
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis
        
        # Preprocess (if your model needs it - ResNet50 typically does)
        img_array = preprocess_input(img_array)
        
        print(f"Image shape: {img_array.shape}")
        
        # Make prediction
        predictions = model.predict(img_array)
        ai_probability = float(predictions[0][0] * 100)
        
        print(f"Chances of being AI generated: {ai_probability}%")
        
        return JSONResponse(content={
            "success": True,
            "ai_probability": round(ai_probability, 2),
            "filename": file.filename
        })
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing image: {str(e)}"}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)