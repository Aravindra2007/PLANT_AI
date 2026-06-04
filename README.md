# 🌱 PLANT AI

An AI-powered plant analysis system that performs **real-time species identification** and **disease detection** from live webcam feeds and uploaded images. Built with deep learning models trained on plant leaf data, it identifies 14 plant species and classifies disease conditions simultaneously.

---

## 🔍 What It Does

Point a camera at a plant leaf — PLANT AI predicts **what plant it is** and **whether it's diseased**, both at the same time, with confidence scores overlaid live on screen. Results above a confidence threshold are automatically logged to a timestamped CSV file.

---

## ✨ Features

- 🎥 **Live webcam detection** — real-time species + disease recognition via `webcam_combined.py`
- 🌿 **14 plant species** recognized: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato
- 🦠 **3 disease states** classified: Healthy, Leaf Spot, Rust
- 📊 **Confidence thresholds** — only logs predictions above 70% (species) and 75% (disease)
- 🗂️ **Auto CSV logging** — high-confidence results written to `plantai_results.csv` every 3 seconds
- ⚡ **FastAPI REST endpoint** — `/predict` accepts image uploads and returns JSON with species + disease + confidence
- 🤖 **Dual-model architecture** — separate Keras models for species (`species_model.keras`) and disease (`disease_model.h5`)

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.x |
| Deep Learning | TensorFlow / Keras |
| Computer Vision | OpenCV |
| REST API | FastAPI |
| Object Detection | Ultralytics (YOLOv8) |
| Data | NumPy, Pandas |
| Dataset Tools | Roboflow, Kaggle |

---

## 📂 Project Structure

```
PLANT_AI/
│
├── app.py                  # FastAPI server — POST /predict endpoint
├── webcam_combined.py      # Live webcam: species + disease detection
├── webcam_species.py       # Live webcam: species detection only
├── webcam_disease.py       # Live webcam: disease detection only
├── species_train.py        # Training script for species model
├── disease_train.py        # Training script for disease model
│
├── species_model.keras     # Trained species classification model
├── disease_model.h5        # Trained disease classification model
│
├── plantai_results.csv     # Auto-generated detection log
├── runs/detect/            # YOLOv8 detection run outputs
│
└── requirements.txt
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Aravindra2007/PLANT_AI.git
cd PLANT_AI
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Option A — Live Webcam (Species + Disease)

Runs both models simultaneously on your webcam feed. Press `Q` to exit.

```bash
python webcam_combined.py
```

The overlay shows species name and disease status with confidence percentages. Detections above the confidence threshold are saved to `plantai_results.csv`.

### Option B — Live Webcam (Species only / Disease only)

```bash
python webcam_species.py
python webcam_disease.py
```

### Option C — REST API

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

Then send a POST request with a plant leaf image:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@leaf.jpg"
```

**Example response:**

```json
{
  "species": "Tomato",
  "species_conf": 94.3,
  "disease": "Leaf_Spot",
  "disease_conf": 88.7
}
```

---

## 🧠 Model Details

Both models take a **224×224 RGB image** normalized to `[0, 1]`.

| Model | Output Classes | Format |
|---|---|---|
| `species_model.keras` | 14 plant species | Keras SavedModel |
| `disease_model.h5` | Healthy, Leaf_Spot, Rust | HDF5 |

To retrain the models on new data:

```bash
python species_train.py
python disease_train.py
```

---

## 📊 CSV Output Format

High-confidence detections are logged automatically:

| Timestamp | Species | Species Confidence (%) | Disease | Disease Confidence (%) |
|---|---|---|---|---|
| 2025-06-01 10:23:45 | Tomato | 94.32 | Leaf_Spot | 88.71 |
| 2025-06-01 10:23:48 | Apple | 91.05 | Healthy | 96.20 |

---

## 🎯 Applications

- Precision agriculture and smart farming
- Early disease detection to reduce crop loss
- Field scouting tools for farmers and agronomists
- Agricultural research and dataset collection
- Greenhouse monitoring systems

---

## 🔮 Future Enhancements

- [ ] Expand disease classes beyond Leaf Spot and Rust
- [ ] Add treatment recommendations per disease
- [ ] Web dashboard for viewing CSV logs
- [ ] Mobile app integration
- [ ] Cloud deployment (Render / Hugging Face Spaces)
- [ ] Support for more plant species

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes
   ```bash
   git commit -m "Add: description of change"
   ```
4. Push and open a Pull Request
   ```bash
   git push origin feature/your-feature
   ```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 👨‍💻 Author

**Aravind Kumar**
GitHub: [@Aravindra2007](https://github.com/Aravindra2007)

---

⭐ If this project was useful, consider giving it a star!
