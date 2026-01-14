import cv2
import numpy as np
import tensorflow as tf
import csv
from datetime import datetime
import os
import time
from tensorflow.keras.models import load_model

SPECIES_THRESHOLD = 70
DISEASE_THRESHOLD = 75


# LOAD MODELS
species_model = load_model("species_model.keras")
disease_model = load_model("disease_model.h5")

CSV_FILE = "plantai_results.csv"

# Create CSV file if not exists
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Timestamp",
            "Species",
            "Species Confidence (%)",
            "Disease",
            "Disease Confidence (%)"
        ])

last_saved_time = 0
SAVE_INTERVAL = 3  # seconds


# CLASS NAMES (must match training folders order)
species_classes = sorted([
    "Apple","Blueberry","Cherry","Corn","Grape",
    "Orange","Peach","Pepper","Potato","Raspberry",
    "Soybean","Squash","Strawberry","Tomato"
])

disease_classes = ["Healthy", "Leaf_Spot", "Rust"]

IMG_SIZE = 224

# PREPROCESS FUNCTION
def preprocess(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# WEBCAM
cap = cv2.VideoCapture(0)

print("📷 Press Q to exit webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = preprocess(frame)

    # PREDICTIONS
    species_pred = species_model.predict(img, verbose=0)
    disease_pred = disease_model.predict(img, verbose=0)

    species_index = np.argmax(species_pred)
    disease_index = np.argmax(disease_pred)

    species = species_classes[species_index]
    disease = disease_classes[disease_index]

    species_conf = species_pred[0][species_index] * 100
    disease_conf = disease_pred[0][disease_index] * 100

    current_time = time.time()



    if (
        species_conf >= SPECIES_THRESHOLD and
        disease_conf >= DISEASE_THRESHOLD and
        current_time - last_saved_time >= SAVE_INTERVAL
    ):
        with open(CSV_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                species,
                f"{species_conf:.2f}",
                disease,
                f"{disease_conf:.2f}"
            ])
        last_saved_time = current_time




    # DISPLAY
    text1 = f"Species: {species} ({species_conf:.2f}%)"
    text2 = f"Disease: {disease} ({disease_conf:.2f}%)"


    cv2.rectangle(frame, (10,10), (520,95), (0,0,0), -1)

    cv2.putText(frame, text1, (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,255,0), 2)

    cv2.putText(frame, text2, (20,75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,255,255), 2)


    cv2.imshow("PlantAI - Species & Disease Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
