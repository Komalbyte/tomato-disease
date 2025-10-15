# importing packages
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
import numpy as np
import uvicorn
import random

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

print("Mock API server started!")

# testing connection
@app.get('/ping')
async def ping():  # asynchronous and non-blocking
    return 'Ready!'

# predicting image (mock prediction)
@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    file_bytes = await file.read()  # preventing blocking
    img = Image.open(BytesIO(file_bytes))  # converting bytes to image
    
    # Mock prediction - randomly select a class with high confidence
    pred_class = random.choice(class_names)
    pred_conf = random.uniform(0.7, 0.95)  # High confidence between 70-95%
    
    print(f"Mock prediction: {pred_class} with confidence {pred_conf:.2f}")
    
    return {
        'pred_class': pred_class,
        'pred_conf': float(pred_conf)
    }

if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)

