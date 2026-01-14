from fastapi import FastAPI, File, UploadFile
import cv2, numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = FastAPI()

species_model = load_model("species_model.keras")
disease_model = load_model("disease_model.h5")

species_classes = [
    "Apple","Blueberry","Cherry","Corn","Grape","Orange",
    "Peach","Pepper","Potato","Raspberry","Soybean",
    "Squash","Strawberry","Tomato"
]
disease_classes = ["Healthy", "Leaf_Spot", "Rust"]

IMG_SIZE = 224

def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = preprocess(img)

    sp = species_model.predict(img)[0]
    ds = disease_model.predict(img)[0]

    return {
        "species": species_classes[int(np.argmax(sp))],
        "species_conf": float(np.max(sp)*100),
        "disease": disease_classes[int(np.argmax(ds))],
        "disease_conf": float(np.max(ds)*100)
    }
