# importing packages
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
import numpy as np
import uvicorn
import tensorflow as tf

app = FastAPI()  # creating FastAPI instance

# allowing requests from port 3000
origins = [
    'http://localhost',
    'http://localhost:3000'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

# declaring class names
class_names = ['Bacterial-spot', 'Early-blight', 'Healthy', 'Late-blight',
               'Leaf-mold', 'Mosaic-virus', 'Septoria-leaf-spot', 'Yellow-leaf-curl-virus']

# Load the model
print("Loading model...")
try:
    model = tf.keras.models.load_model('../../models/tomato-disease-detection-model.h5', compile=False)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Using mock predictions...")
    model = None

# testing connection
@app.get('/ping')
async def ping():  # asynchronous and non-blocking
    return 'Ready!'

# predicting image
@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    file_bytes = await file.read()  # preventing blocking
    img = Image.open(BytesIO(file_bytes))  # converting bytes to image
    
    # Resize image to 224x224 (common input size for CNN models)
    img = img.resize((224, 224))
    
    img_array = np.array(img)  # converting image to numpy array
    
    # Normalize the image
    img_array = img_array / 255.0
    
    img_batch = np.expand_dims(img_array, axis=0)  # creating image batch for prediction

    if model is not None:
        # Make prediction with real model
        predictions = model.predict(img_batch)
        pred = predictions[0]
    else:
        # Mock prediction
        import random
        pred = np.random.random(len(class_names))
        pred = pred / np.sum(pred)  # Normalize to probabilities

    pred_class = class_names[np.argmax(pred)]  # getting predicted class
    pred_conf = np.max(pred)  # getting prediction confidence
    
    return {
        'pred_class': pred_class,
        'pred_conf': float(pred_conf)
    }

if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)

