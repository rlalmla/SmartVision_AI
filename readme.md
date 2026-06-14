# 🤖 SmartVision_AI

SmartVision_AI is an end-to-end **Computer Vision application** built using **Deep Learning, Transfer Learning, YOLOv8m, and Streamlit**.

The project performs both:

* **Image Classification** using multiple CNN models
* **Object Detection** using YOLOv8m

The application enables users to upload images and compare predictions from multiple classification models while also performing real-time object detection.

---

# 📌 Features

## 🖼️ Image Classification

Compare predictions from **4 CNN Transfer Learning models**:

* **ResNet50** ⭐ *(Best Performing Model)*
* **VGG16**
* **MobileNetV2** ⚡ *(Fastest & Smallest Model)*
* **EfficientNetB0**

### Classification Features

✅ Upload custom image for prediction
✅ Multi-model comparison
✅ Confidence score display
✅ Side-by-side prediction results
✅ Performance comparison dashboard

---

## 🎯 Object Detection

Object detection powered by **YOLOv8m**.

### Detection Features

✅ Multi-object detection
✅ Bounding boxes with labels
✅ Confidence score visualization
✅ Real-world object localization

---

## 📊 Model Performance Dashboard

Dedicated performance analysis page including:

* Classification metrics comparison
* Accuracy, Precision, Recall, F1 Score
* Confusion matrices
* YOLOv8m performance evaluation
* Model comparison visualizations

---

# 🧠 Models Used

## Image Classification Models

| Model          | Accuracy   | Precision  | Recall     | F1 Score   | Model Size    |
| -------------- | ---------- | ---------- | ---------- | ---------- | ------------- |
| **ResNet50** ⭐ | **55.25%** | **57.64%** | **55.25%** | **55.13%** | **164.60 MB** |
| VGG16          | 38.75%     | 43.06%     | 38.75%     | 35.36%     | 204.80 MB     |
| MobileNetV2 ⚡  | 49.75%     | 48.18%     | 49.75%     | 48.10%     | 13.14 MB      |
| EfficientNetB0 | 49.50%     | 53.94%     | 49.50%     | 49.03%     | 25.08 MB      |

### 🏆 Best Classification Model

**ResNet50** achieved the highest classification accuracy (**55.25%**).

### ⚡ Fastest & Smallest Model

**MobileNetV2** demonstrated efficient lightweight performance (**13.14 MB**).

---

## 🎯 Object Detection Model

### YOLOv8m

Used for multi-object detection with optimized inference and localization.

---

# 🏷️ Supported Categories

The project supports the following **26 COCO object categories**:

```text
Airplane | Bed | Bench | Bicycle | Bird | Bottle | Bowl
Bus | Cake | Car | Cat | Chair | Couch | Cow
Cup | Dog | Elephant | Horse | Motorcycle
Person | Pizza | Potted Plant | Stop Sign
Traffic Light | Train | Truck
```

---

# 📂 Dataset Acquisition & Preprocessing

## Dataset Used

This project uses a **curated subset of the COCO dataset** for optimal training efficiency.

Since the complete COCO dataset contains **122K+ images**, this project uses a focused subset to improve transfer learning efficiency and reduce training time.

### Dataset Statistics

* **2,500 total images**
* **100 images per class**
* **26 selected categories**
* **Train / Validation / Test split**

  * **70% Training**
  * **15% Validation**
  * **15% Testing**

---

## Dataset Preparation Pipeline

The preprocessing pipeline includes:

✅ COCO dataset loading from Hugging Face (streaming mode)
✅ Automated class filtering
✅ Train/validation/test split creation
✅ Classification image generation (**224×224 crops**)
✅ YOLO annotation preparation
✅ Metadata generation
✅ Organized folder creation for classification and detection tasks

---

# 🗂️ Project Structure

```text
SmartVision_AI/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── assets/
│   └── Demo images
│
├── smartvision_dataset/
│   └── results/
│       ├── models/
│       │   ├── ResNet50_best.h5
│       │   ├── VGG16_best.h5
│       │   ├── MobileNetV2_best.h5
│       │   ├── EfficientNetB0_best.h5
│       │   └── best.pt
│       │
│       ├── confusion matrices
│       ├── performance charts
│       └── reports
├── notebooks/
│   ├── smartvision_project.ipynb
│   ├── eda.ipynb
│   ├── classification_training.ipynb
│   └── yolo_detection.ipynb
```

---

# ⚙️ Installation & Setup

## Step 1: Clone Repository

```bash
git clone <your-github-repo-link>
cd SmartVision_AI
```

---

## Step 2: Install Git LFS

This project uses **Git LFS (Large File Storage)** for model files (`.h5`, `.pt`).

Install Git LFS:

```bash
git lfs install
```

Download model files:

```bash
git lfs pull
```

---

## Step 3: Create Virtual Environment (venv)

Create virtual environment:

```bash
python -m venv venv
```

Activate virtual environment:

### Windows

```bash
venv\Scripts\activate
```

After activation, terminal should show:

```text
(venv)
```

---

## Step 4: Install Dependencies

Install required libraries:

```bash
pip install -r requirements.txt
```

---

## Step 5: Run SmartVision_AI

Launch the Streamlit application:

```bash
python -m streamlit run app.py
```

Application will open at:

```text
http://localhost:8501
```

---

# 📒 Notebook Workflow (Reference Only)

The notebooks included in this repository document the **complete development workflow and methodology**.

⚠️ **Running notebooks is NOT required to use the application.**

Pre-trained models are already included via **Git LFS**, so the application works directly after setup.

---

### 1️⃣ `notebooks/smartvision_project.ipynb`

**Purpose:** Dataset acquisition & preprocessing

This notebook:

* Loads COCO dataset
* Creates curated subset
* Prepares train/validation/test splits
* Generates metadata
* Organizes classification & detection folders

---

### 2️⃣ `notebooks/eda.ipynb`

**Purpose:** Exploratory Data Analysis (EDA)

This notebook:

* Analyzes dataset distribution
* Visualizes class balance
* Generates charts and insights

---

### 3️⃣ `notebooks/classification_training.ipynb`

**Purpose:** CNN classification training

This notebook:

* Trains:

  * ResNet50
  * VGG16
  * MobileNetV2
  * EfficientNetB0
* Evaluates model performance
* Saves trained `.h5` models

---

### 4️⃣ `notebooks/yolo_detection.ipynb`

**Purpose:** YOLOv8m object detection training

This notebook:

* Trains YOLOv8m
* Evaluates object detection performance
* Generates confusion matrices
* Saves trained `.pt` model

---

### 5️⃣ `app.py`

**Purpose:** Run SmartVision_AI

Launch app:

```bash
python -m streamlit run app.py
```

---

# 📦 Main Libraries Used

```text
streamlit==1.28.0
tensorflow==2.16.1
ultralytics==8.0.200
numpy==1.24.3
pandas==2.0.3
Pillow==10.0.0
matplotlib==3.7.2
opencv-python-headless==4.8.1.78
torch==2.0.1
torchvision==0.15.2
scikit-learn==1.3.0
scipy==1.10.1
```

---

# ⚠️ Troubleshooting

## TensorFlow/Keras Model Loading Issue

Use:

```text
tensorflow==2.16.1
```

Do **not install standalone `keras` separately**, since TensorFlow includes Keras internally.

---

## OpenCV (`cv2`) Error

Install:

```bash
pip install opencv-python-headless
```

---

## Streamlit Launch Issue

Always run:

```bash
python -m streamlit run app.py
```

instead of:

```bash
streamlit run app.py
```

to ensure the correct virtual environment is used.

---

# 🚀 Future Improvements

* Hugging Face deployment
* Real-time webcam detection
* More object categories
* Improved classification accuracy
* Video inference support
* Cloud optimization

---

# 👨‍💻 Author

**Rama Naren**

Built using **Deep Learning, Transfer Learning, YOLOv8m, TensorFlow, PyTorch, and Streamlit**.
