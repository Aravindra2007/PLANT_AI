import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# -------------------------
# Load trained model
# -------------------------
model = load_model("disease_model.h5")

# -------------------------
# Class labels (ORDER MUST MATCH TRAINING)
# -------------------------
class_names = ["Healthy", "Leaf_Spot", "Rust"]  # adjust if needed

# -------------------------
# Open webcam
# -------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam not accessible")
    exit()

print("✅ Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for model
    img = cv2.resize(frame, (224, 224))
    img = img.astype("float") / 255.0
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    # Predict
    preds = model.predict(img, verbose=0)[0]
    class_id = np.argmax(preds)
    confidence = preds[class_id]

    label = f"{class_names[class_id]} ({confidence*100:.1f}%)"

    # Display
    cv2.putText(
        frame, label, (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1,
        (0, 255, 0), 2
    )

    cv2.imshow("Plant Disease Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

