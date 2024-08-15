import numpy as np

from fastapi import FastAPI, Body
from fastapi.staticfiles import StaticFiles
from myapp.model import Model

import pickle
import os

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# load model
model = Model()

# app
app = FastAPI(title='Symbol detection', docs_url='/docs')

# api
@app.post('/api/predict')
def predict(image: str = Body(..., description='image pixels list')):
    image = np.array(list(map(int, image[1:-1].split(','))))
    
    '''image = image.reshape(28, 28)
    image = cv2.blur(image,(3,3))
    image = image.flatten()'''
    
    pred = model.predict(image)
    return {'prediction': pred}

# static files
app.mount('/', StaticFiles(directory='static', html=True), name='static')
