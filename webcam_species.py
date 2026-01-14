import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained species model
# model = load_model("species_model.h5")  # change name if needed
model = load_model("species_model.keras")


# Class names (must match folder names)
class_names = [
    'Apple', 'Blueberry', 'Cherry', 'Corn', 'Orange', 'Peach',
    'Pepper', 'Potato', 'Raspberry', 'Soybean',
    'Squash', 'Strawberry', 'Tomato', 'Grape___healthy'
]

IMG_SIZE = 224

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    preds = model.predict(img, verbose=0)
    class_id = np.argmax(preds)
    confidence = preds[0][class_id]

    label = f"{class_names[class_id]} ({confidence*100:.1f}%)"

    # Display
    cv2.putText(frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    cv2.imshow("Plant Species Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



