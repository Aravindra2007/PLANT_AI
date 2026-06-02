# 🌱 PLANT_AI

An AI-powered plant disease detection system that helps farmers, gardeners, and agricultural researchers identify plant diseases from leaf images using Machine Learning and Computer Vision.

## 📖 Overview

PLANT_AI is designed to detect and classify plant diseases from uploaded leaf images. The system leverages deep learning models trained on plant disease datasets to provide fast and accurate predictions, enabling early disease detection and better crop management.

By identifying diseases at an early stage, farmers can reduce crop losses, improve productivity, and make informed treatment decisions.

---

## 🚀 Features

* 🌿 Plant disease detection from leaf images
* 🤖 Deep Learning-based image classification
* 📸 Image upload and prediction interface
* 📊 High-accuracy disease recognition
* ⚡ Fast inference and real-time results
* 🌍 User-friendly application for farmers and researchers
* 📱 Responsive design for desktop and mobile devices

---

## 🛠️ Technology Stack

### Frontend

* HTML
* CSS
* JavaScript

### Backend

* Python
* Flask

### Machine Learning

* TensorFlow / Keras
* OpenCV
* NumPy
* Pandas

### Dataset

* Plant Disease Dataset
* PlantVillage Dataset (or equivalent)

---

## 📂 Project Structure

```text
PLANT_AI/
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── model/
│   └── trained_model.h5
│
├── uploads/
│
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Aravindra2007/PLANT_AI.git
cd PLANT_AI
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate Environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux/Mac**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python app.py
```

### 5. Open Browser

```text
http://localhost:5000
```

---

## 🌿 How It Works

1. Upload a plant leaf image.
2. The image is preprocessed using OpenCV.
3. The trained deep learning model analyzes the image.
4. The system predicts the disease category.
5. Results are displayed to the user.

---

## 📈 Applications

* Smart Farming
* Precision Agriculture
* Crop Health Monitoring
* Agricultural Research
* Disease Prevention Systems

---

## 🎯 Future Enhancements

* Real-time camera detection
* Disease severity estimation
* Treatment recommendations
* Multi-language support
* Mobile application
* Cloud deployment
* IoT integration for smart farming

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch

```bash
git checkout -b feature-name
```

3. Commit changes

```bash
git commit -m "Added new feature"
```

4. Push changes

```bash
git push origin feature-name
```

5. Open a Pull Request

---

## 👨‍💻 Author

**Aravind Kumar**

GitHub: https://github.com/Aravindra2007

---

## 📜 License

This project is licensed under the MIT License.

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub. It helps the project reach more developers and researchers.
