import streamlit as st
import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from ultralytics import YOLO
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="SmartVision AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS FOR BETTER UI
# ============================================================================

st.markdown("""
<style>
            
/* ==========================================================
   PREMIUM HOVER HEADER - FIXED
========================================================== */

.page-header {
    background: linear-gradient(
        135deg,
        rgba(76, 81, 191, 0.25),
        rgba(139, 92, 246, 0.20)
    );

    padding: 32px 30px;

    border-radius: 22px;

    border: 1px solid rgba(255,255,255,0.12);

    backdrop-filter: blur(12px);

    transition: all 0.35s ease;

    box-shadow:
        0 4px 20px rgba(0,0,0,0.15);

    margin-top: 25px;
    margin-bottom: 25px;

    display: flex;
    flex-direction: column;
    justify-content: center;
}

/* Hover Effect */
.page-header:hover {
    transform: translateY(-4px);

    box-shadow:
        0 10px 30px rgba(139, 92, 246, 0.35),
        0 0 20px rgba(76, 81, 191, 0.25);

    border: 1px solid rgba(139, 92, 246, 0.4);
}

/* Main Heading */
.page-title {
    font-size: 38px;
    font-weight: 700;
    color: white;
    margin: 0;
    line-height: 1.2;
}

/* Subtitle */
.page-subtitle {
    font-size: 16px;
    color: #e5e7eb;
    margin-top: 8px;
}
            
/* ==========================================================
TOP LEFT WHITE PATCH
========================================================== */

[data-testid="stSidebar"]::before {
    content: "";
    display: block;
    height: 58px;   /* adjust if needed */
    background: white;
    margin-top: -1px;
}
            
/* ==========================================================
MAIN APP BACKGROUND
========================================================== */

.stApp {
    background-color: #f6f8fc;
}

/* ==========================================================
MAIN PAGE SPACING
========================================================== */

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* ==========================================================
SIDEBAR
========================================================== */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #1e293b,
        #0f172a
    );
    color: white;
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Top Left White Alignment Patch */

[data-testid="stSidebar"]::before {
    content: "";
    display: block;
    height: 58px;
    background: white;
    margin-top: -1px;
}

/* Sidebar Text */

[data-testid="stSidebar"] * {
    color: white !important;
}

/* Sidebar Radio Buttons */

[data-testid="stSidebar"] .stRadio label {
    padding: 10px 12px;
    border-radius: 12px;
    transition: 0.3s ease;
}

/* Hover Effect */

[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.08);
    transform: translateX(4px);
}

/* Sidebar Card */

.sidebar-card {
    background: linear-gradient(
        135deg,
        rgba(37,99,235,0.18),
        rgba(124,58,237,0.12)
    );
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 18px;
    margin-top: 20px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
}
/* ==========================================================
HEADERS
========================================================== */

.header-title {
    font-size: 38px;
    font-weight: 700;
    text-align: center;
    color: #2563eb;
    margin-bottom: 20px;
    padding: 18px;
    border-radius: 18px;

    background: linear-gradient(
        90deg,
        #dbeafe,
        #ede9fe
    );

    box-shadow:
        0px 4px 12px rgba(0,0,0,0.08);
}

/* ==========================================================
INFO BOX
========================================================== */

.info-box {

    background: linear-gradient(
        90deg,
        #eff6ff,
        #eef2ff
    );

    border-left: 6px solid #3b82f6;

    color: #1e293b;

    padding: 18px;
    border-radius: 14px;
    margin-bottom: 20px;
    font-size: 16px;

    box-shadow:
        0px 2px 6px rgba(0,0,0,0.06);
}

/* ==========================================================
SUCCESS BOX
========================================================== */

.success-box {

    background: #ecfdf5;

    border-left: 6px solid #10b981;

    color: #065f46;

    padding: 18px;
    border-radius: 14px;
    margin-bottom: 20px;

    box-shadow:
        0px 2px 6px rgba(0,0,0,0.06);
}

/* ==========================================================
METRIC CARDS
========================================================== */

[data-testid="metric-container"] {

    background: white;

    border-radius: 18px;

    padding: 15px;

    box-shadow:
        0px 4px 12px rgba(0,0,0,0.08);

    border: 1px solid #e5e7eb;
}

[data-testid="stMetricValue"] {
    font-size: 28px;
    font-weight: bold;
    color: #2563eb;
}

[data-testid="stMetricLabel"] {
    color: #475569;
    font-weight: 600;
}

/* ==========================================================
BUTTONS
========================================================== */

.stButton > button {

    background: linear-gradient(
        90deg,
        #2563eb,
        #7c3aed
    );

    color: white;

    border-radius: 12px;

    border: none;

    padding: 10px 18px;

    font-weight: 600;
}

.stButton > button:hover {

    background: linear-gradient(
        90deg,
        #1d4ed8,
        #6d28d9
    );

    color: white;
}

/* ==========================================================
UPLOAD BOX
========================================================== */

[data-testid="stFileUploader"] {

    background: white;

    border-radius: 15px;

    padding: 10px;

    border: 2px dashed #cbd5e1;
}

/* ==========================================================
TABLES / DATAFRAME
========================================================== */

[data-testid="stDataFrame"] {

    border-radius: 15px;

    overflow: hidden;

    border: 1px solid #e2e8f0;
}

/* ==========================================================
HIDE FOOTER
========================================================== */

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)



# ============================================================================
# LOAD CLASSIFICATION MODELS
# ============================================================================

@st.cache_resource
def load_classification_models():

    try:

        models = {

            "ResNet50":
            keras.models.load_model(
                "smartvision_dataset/results/models/ResNet50_best.h5",
                 compile=False
            ),

            "VGG16":
            keras.models.load_model(
                "smartvision_dataset/results/models/VGG16_best.h5",
                compile=False
            ),

            "MobileNetV2":
            keras.models.load_model(
                "smartvision_dataset/results/models/MobileNetV2_best.h5",
                compile=False
            ),

            "EfficientNetB0":
            keras.models.load_model(
                "smartvision_dataset/results/models/EfficientNetB0_best.h5",
                compile=False
            )

        }

        return models

    except Exception as e:

        st.error(
            f"❌ Error loading classification models: {str(e)}"
        )

        return None

@st.cache_resource
def load_yolo_model():
    """Load YOLOv8m object detection model with error handling."""
    try:
        model_path = "smartvision_dataset/results/models/best.pt"
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading YOLOv8 model: {str(e)}")
        return None


# ============================================================================
# CLASS LABELS FOR CLASSIFICATION
# ============================================================================
CLASS_LABELS = [
    'airplane', 'bed', 'bench', 'bicycle', 'bird', 'bottle', 'bowl', 'bus',
    'cake', 'car', 'cat', 'chair', 'couch', 'cow', 'cup', 'dog',
    'elephant', 'horse', 'motorcycle', 'person', 'pizza', 'potted plant',
    'stop sign', 'traffic light', 'train', 'truck'
]
# ============================================================================
# MODEL PREPROCESSING
# ============================================================================

def preprocess_image(
    image,
    model_name
):

    img_array = np.array(
        image
    )

    img_resized = cv2.resize(
        img_array,
        (224, 224)
    )

    img_resized = img_resized.astype(
        np.float32
    )

    img_batch = np.expand_dims(
        img_resized,
        axis=0
    )

    if model_name == "ResNet50":

        return tf.keras.applications.resnet50.preprocess_input(
            img_batch
        )

    elif model_name == "VGG16":

        return tf.keras.applications.vgg16.preprocess_input(
            img_batch
        )

    elif model_name == "MobileNetV2":

        return tf.keras.applications.mobilenet_v2.preprocess_input(
            img_batch
        )

    elif model_name == "EfficientNetB0":

        return tf.keras.applications.efficientnet.preprocess_input(
            img_batch
        )

    return img_batch

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
st.sidebar.title("🤖 SmartVision AI")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "📍 Navigate to Page:",
    [
        "🏠 Home",
        "🖼️ Image Classification",
        "🎯 Object Detection",
        "📊 Model Performance",
        "ℹ️ About Project"
    ],
    label_visibility="visible"
)

st.sidebar.markdown("---")
st.sidebar.info(
    "🚀 **SmartVision AI** - An intelligent computer vision system powered by "
    "YOLOv8m and ResNet50 for image classification and object detection."
)

# ============================================================================
# PAGE ROUTING
# ============================================================================
# ============================================================================
# HOME PAGE
# ============================================================================

if page == "🏠 Home":

    st.markdown("""
    <div class="page-header">
        <div class="page-title">
            🤖 SmartVision AI
        </div>
        <div class="page-subtitle">
            Intelligent Computer Vision System
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ========================================================================
    # PROJECT OVERVIEW
    # ========================================================================

    st.markdown(
        ("## Discover AI-Powered Visual Intelligence")
    )

    st.write(
        """
        SmartVision AI is an advanced computer vision system
        designed to analyze and understand images using
        deep learning models.

        The system supports:

        ✅ Image Classification  
        ✅ Object Detection  
        ✅ Multi-Object Recognition  
        ✅ AI-Powered Visual Analysis
        """
    )

    st.markdown("---")

    # ========================================================================
    # KEY FEATURES
    # ========================================================================

    st.markdown("## ✨ Key Features")

    feature_col1, feature_col2 = st.columns(2)

    with feature_col1:

        st.info("""
### 🖼️ Image Classification

- Classify images into 26 categories
- Powered by ResNet50
- Instant predictions with confidence score
        """)

        st.info("""
### 🎯 Object Detection

- Detect multiple objects
- Real-time bounding boxes
- YOLOv8m powered detection
        """)

    with feature_col2:

        st.info("""
### 🔍 Multi-Object Recognition

- Detect all objects in image
- Confidence scores for each object
- Detailed visual understanding
        """)

        st.info("""
### 🧠 AI-Powered Visual Analysis

- Deep learning powered
- Fast inference
- Professional computer vision
        """)

    st.markdown("---")

    # ========================================================================
    # MODELS USED
    # ========================================================================

    st.markdown("## 🚀 Models Used")

    model_col1, model_col2 = st.columns(2)

    with model_col1:

        st.success("""
### 🧠 ResNet50

**Purpose:** Image Classification

- Deep residual neural network
- Predicts image category
- Supports 26 classes
        """)

    with model_col2:

        st.success("""
### 🎯 YOLOv8m

**Purpose:** Object Detection

- Detects multiple objects
- Generates bounding boxes
- Real-time localization
        """)

    st.markdown("---")

    # ========================================================================
    # SAMPLE DEMO IMAGES
    # ========================================================================

    st.markdown("## 🖼️ Sample Demo Images")

    st.write(
        "Use these example images to test SmartVision AI."
    )

    demo_col1, demo_col2 = st.columns(2)

    with demo_col1:

        st.image(
            "assets/dogs_demo.jpg",
            caption="🐶 Dog Classification Demo",
            use_column_width=True
        )

        st.image(
            "assets/multi_object_demo.jpg",
            caption="🚦 Multi-Object Detection Demo",
            use_column_width=True
        )

    with demo_col2:

        st.image(
            "assets/horse_demo.jpg",
            caption="🐎 Horse Classification Demo",
            use_column_width=True
        )

    
        st.image(
            "assets/cup_demo.jpg",
            caption="🥤 Cup Classification Demo",
            use_column_width=True
        )

        st.image(
            "assets/potted_plant_demo.jpg",
            caption="🌱 Potted Plant Classification Demo",
            use_column_width=True
        )


    st.markdown("---")

    # ========================================================================
    # HOW TO USE
    # ========================================================================

    st.markdown("## 📌 How to Use")

    instruction_col1, instruction_col2 = st.columns(2)

    with instruction_col1:

        st.info("""
### 🖼️ Image Classification

1. Open **Image Classification**
2. Upload a **single-object image**
3. View prediction and confidence score
        """)

    with instruction_col2:

        st.info("""
### 🎯 Object Detection

1. Open **Object Detection**
2. Upload **multi-object image**
3. View bounding boxes and detections
        """)

    st.markdown("---")

    # ========================================================================
    # SYSTEM STATUS
    # ========================================================================

    st.markdown("## 📊 System Status")

    status_col1, status_col2, status_col3 = st.columns(3)

    with status_col1:

        st.metric(
            label="🧠 ResNet50 Model",
            value="Ready ✅"
        )

    with status_col2:

        st.metric(
            label="🎯 YOLOv8m Model",
            value="Ready ✅"
        )

    with status_col3:

        st.metric(
            label="⚡ System Status",
            value="Online ✅"
        )

    st.success(
        "✅ System is ready for inference! Navigate to Image Classification or Object Detection to get started."
    )

    st.markdown("---")

    # ========================================================================
    # FOOTER
    # ========================================================================

    st.markdown(
        """
        <div style="text-align:center;">
        <h4>SmartVision AI v1.0</h4>
        <p>Built using Streamlit, TensorFlow and YOLOv8</p>
        </div>
        """,
        unsafe_allow_html=True
    )

elif page == "🖼️ Image Classification":

    # ========================================================================
    # IMAGE CLASSIFICATION PAGE
    # ========================================================================

    st.markdown("""
        <div class="page-header">
            <div class="page-title">
                🖼️ Image Classification
            </div>
            <div class="page-subtitle">
                Analyze and classify images using deep learning models
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(
        '''
        <div class="info-box">
        <b>
        Compare predictions from 4 CNN models:
        ResNet50, VGG16, MobileNetV2, EfficientNetB0
        </b>
        </div>
        ''',
        unsafe_allow_html=True
    )

    st.markdown("### 📸 Upload Your Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:

        classification_models = (
            load_classification_models()
        )

        if classification_models is None:

            st.error(
                "❌ Failed to load classification models."
            )

        else:

            try:

                image = Image.open(
                    uploaded_file
                ).convert("RGB")

                col1, col2 = st.columns(
                    [1, 1.2]
                )

                # ============================================================
                # LEFT SIDE - IMAGE
                # ============================================================

                with col1:

                    st.markdown(
                        "#### 📷 Uploaded Image"
                    )

                    st.image(
                        image,
                        use_column_width=True,
                        caption="Uploaded Image"
                    )

                # ============================================================
                # RIGHT SIDE - RESULTS
                # ============================================================

                with col2:

                    st.markdown(
                        "#### 🤖 Model Comparison"
                    )

                    with st.spinner(
                        "🔄 Running classification..."
                    ):

                        model_results = {}

                        for model_name, model in (
                            classification_models.items()
                        ):

                            processed_image = (
                                preprocess_image(
                                    image,
                                    model_name
                                )
                            )

                            prediction = (
                                model.predict(
                                    processed_image,
                                    verbose=0
                                )
                            )

                            pred_idx = np.argmax(
                                prediction[0]
                            )

                            confidence = float(
                                np.max(
                                    prediction[0]
                                )
                            )

                            predicted_class = (
                                CLASS_LABELS[
                                    pred_idx
                                ]
                            )

                            model_results[
                                model_name
                            ] = {

                                "prediction":
                                predicted_class,

                                "confidence":
                                confidence,

                                "full_prediction":
                                prediction[0]
                            }

                        st.success(
                            "✅ Classification Complete!"
                        )

                        # ====================================================
                        # MODEL COMPARISON
                        # ====================================================

                        model_cols = st.columns(4)

                        model_names = list(
                            model_results.keys()
                        )

                        for i, model_name in enumerate(
                            model_names
                        ):

                            with model_cols[i]:

                                st.metric(
                                    label=model_name,
                                    value=model_results[
                                        model_name
                                    ][
                                        "prediction"
                                    ].title(),

                                    delta=f"{model_results[model_name]['confidence'] * 100:.2f}%"
                                )

                        st.markdown("---")

                        # ====================================================
                        # DROPDOWN FOR TOP 5
                        # ====================================================

                        selected_model = st.selectbox(
                            "📌 Select Model for Top-5 Predictions",
                            model_names
                        )

                        selected_prediction = (
                            model_results[
                                selected_model
                            ][
                                "full_prediction"
                            ]
                        )

                        top_5_indices = np.argsort(
                            selected_prediction
                        )[-5:][::-1]

                        prediction_data = []

                        for idx in top_5_indices:

                            prediction_data.append(
                                {
                                    "Class":
                                    CLASS_LABELS[
                                        idx
                                    ].title(),

                                    "Confidence (%)":
                                    f"{selected_prediction[idx] * 100:.2f}"
                                }
                            )

                        st.markdown(
                            f"### 📈 Top-5 Predictions ({selected_model})"
                        )

                        st.dataframe(
                            prediction_data,
                            width=800
                        )

            except Exception as e:

                st.error(
                    f"❌ Error: {str(e)}"
                )

    else:

        st.info(
            "📁 Please upload an image to begin classification."
        )

        st.markdown("---")

        st.markdown(
            """
### 📝 Supported Categories

Airplane | Bed | Bench | Bicycle | Bird | Bottle | Bowl | Bus | Cake | Car | Cat | Chair | Couch | Cow | Cup | Dog | Elephant | Horse | Motorcycle | Person | Pizza | Potted Plant | Stop Sign | Traffic Light | Train | Truck
            """
        )
elif page == "🎯 Object Detection":

    # ========================================================================
    # OBJECT DETECTION PAGE
    # ========================================================================
    st.markdown("""
        <div class="page-header">
            <div class="page-title">
                🎯 Object Detection
            </div>
            <div class="page-subtitle">
                 Detect and localize multiple objects using YOLOv8m
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(
        '''
        <div class="info-box">
        <b>
        Detect multiple objects using YOLOv8m with
        real-time bounding boxes
        </b>
        </div>
        ''',
        unsafe_allow_html=True
    )

    st.markdown("### 📸 Upload Your Image")

    st.markdown(
        "Upload a JPG, JPEG, or PNG image "
        "to detect objects instantly."
    )

    # ============================================================
    # IMAGE UPLOAD
    # ============================================================

    uploaded_file = st.file_uploader(
        label="Choose an image file",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    # ============================================================
    # CONFIDENCE THRESHOLD
    # ============================================================

    confidence_threshold = st.slider(
        "🎚️ Detection Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05
    )

    st.info(
        f"Current Threshold: "
        f"{confidence_threshold:.2f}"
    )

    # ============================================================
    # RUN DETECTION
    # ============================================================

    if uploaded_file is not None:

        yolo_model = load_yolo_model()

        if yolo_model is None:

            st.error(
                "❌ Failed to load YOLOv8 model."
            )

        else:

            try:

                # Load image
                image = Image.open(
                    uploaded_file
                ).convert("RGB")

                img_array = np.array(
                    image
                )

                # ====================================================
                # YOLO DETECTION
                # ====================================================

                with st.spinner(
                    "🔄 Detecting objects..."
                ):

                    results = yolo_model.predict(
                        source=img_array,
                        conf=float(
                            confidence_threshold
                        )
                    )

                # ====================================================
                # ANNOTATED IMAGE
                # ====================================================

                annotated_img = (
                    results[0].plot()
                )

                annotated_img_rgb = (
                    cv2.cvtColor(
                        annotated_img,
                        cv2.COLOR_BGR2RGB
                    )
                )

                annotated_pil = (
                    Image.fromarray(
                        annotated_img_rgb
                    )
                )

                # ====================================================
                # DISPLAY IMAGES
                # ====================================================

                col1, col2 = st.columns(2)

                with col1:

                    st.markdown(
                        "#### 📷 Uploaded Image"
                    )

                    st.image(
                        image,
                        width=350,
                        caption="Original Image"
                    )

                with col2:

                    st.markdown(
                        "#### 🎯 Detection Results"
                    )

                    st.image(
                        annotated_pil,
                        width=350,
                        caption="Annotated with Detections"
                    )

                st.markdown("---")

                # ====================================================
                # DETECTION RESULTS
                # ====================================================

                detections = (
                    results[0].boxes
                )

                if len(detections) > 0:

                    st.success(
                        f"✅ Detected "
                        f"{len(detections)} "
                        f"object(s)!"
                    )

                    st.markdown(
                        "#### 📊 Detected Objects"
                    )

                    detection_list = []

                    for box in detections:

                        class_id = int(
                            box.cls[0]
                        )

                        confidence = float(
                            box.conf[0]
                        )

                        class_name = (
                            results[0]
                            .names[class_id]
                        )

                        detection_list.append(
                            {
                                "Object Name":
                                class_name.title(),

                                "Confidence Score":
                                f"{confidence * 100:.2f}%"
                            }
                        )

                    detection_df = (
                        pd.DataFrame(
                            detection_list
                        )
                    )

                    st.dataframe(
                        detection_df,
                        width=800
                    )

                    st.markdown("---")

                    # ====================================================
                    # DETECTION SUMMARY
                    # ====================================================

                    st.markdown(
                        "#### 📈 Detection Summary"
                    )

                    summary_col1, summary_col2, summary_col3 = st.columns(3)

                    with summary_col1:

                        st.metric(
                            label="🎯 Total Objects",
                            value=len(
                                detections
                            )
                        )

                    with summary_col2:

                        avg_confidence = (
                            np.mean(
                                [
                                    float(
                                        box.conf[0]
                                    )
                                    for box in detections
                                ]
                            )
                        )

                        st.metric(
                            label="📊 Avg Confidence",
                            value=f"{avg_confidence * 100:.2f}%"
                        )

                    with summary_col3:

                        unique_classes = len(
                            set(
                                [
                                    results[0].names[
                                        int(box.cls[0])
                                    ]
                                    for box in detections
                                ]
                            )
                        )

                        st.metric(
                            label="🏷️ Unique Classes",
                            value=unique_classes
                        )

                else:

                    st.warning(
                        "⚠️ No objects detected."
                    )

            except Exception as e:

                st.error(
                    f"❌ Error during object detection: {str(e)}"
                )

    else:

        st.info(
            "📁 Please upload an image file "
            "to detect objects."
        )
        st.markdown("---")
        st.markdown("""
        ### 📝 About YOLOv8m Object Detection
        
        **YOLOv8m** is a state-of-the-art object detection model that can detect hundreds of different object classes.
        
        **Key Features:**
        - ✅ Real-time inference speed
        - ✅ High accuracy with bounding boxes
        - ✅ Detects multiple objects in a single image
        - ✅ Returns confidence scores for each detection
        - ✅ Professional-grade computer vision
        
        **How it works:**
        1. Upload your image (JPG, JPEG, or PNG)
        2. YOLOv8m analyzes the image
        3. Detects all objects and draws bounding boxes
        4. Shows confidence scores for each detection
        5. Displays results in a professional table
        """)
elif page == "📊 Model Performance":
    # ========================================================================
    # MODEL PERFORMANCE PAGE
    # ========================================================================
    st.markdown("""
    <div class="page-header">
        <div class="page-title">
            📊 Model Performance
        </div>
        <div class="page-subtitle">
            Evaluate model accuracy, metrics, and performance insights
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ========================================================================
    # SECTION 1: PERFORMANCE OVERVIEW DASHBOARD
    # ========================================================================
    st.markdown("## 🎯 Performance Overview")
    
    # Load model comparison data
    comparison_csv_path = "smartvision_dataset/results/model_comparison.csv"
    
    if os.path.exists(comparison_csv_path):
        comparison_df = pd.read_csv(comparison_csv_path)
        
        # Get best values
        best_accuracy_idx = comparison_df['Accuracy'].idxmax()
        best_accuracy_model = comparison_df.loc[best_accuracy_idx, 'Model']
        best_accuracy_value = comparison_df.loc[best_accuracy_idx, 'Accuracy']
        
        smallest_model_idx = comparison_df['Model Size (MB)'].idxmin()
        smallest_model_name = comparison_df.loc[smallest_model_idx, 'Model']
        smallest_model_size = comparison_df.loc[smallest_model_idx, 'Model Size (MB)']
        
        # Create metric cards
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric(
                label="🏆 Best Classification",
                value=best_accuracy_model,
                delta=f"{best_accuracy_value * 100:.2f}% Accuracy"
            )
        
        with metric_col2:
            st.metric(
                label="📦 Smallest Model",
                value=smallest_model_name,
                delta=f"{smallest_model_size:.2f} MB"
            )
        
        with metric_col3:
            st.metric(
                label="⚡ Fastest Model",
                value="MobileNetV2",
                delta="Lightweight Architecture"
            )
        
        with metric_col4:
            st.metric(
                label="🎯 Detection Model",
                value="YOLOv8m",
                delta="Final Refinement"
            )
    
    st.markdown("---")
    
    # ========================================================================
    # SECTION 2: MODEL COMPARISON DASHBOARD
    # ========================================================================
    st.markdown("## 📊 Classification Models Comparison")
    
    if os.path.exists(comparison_csv_path):
        comparison_df = pd.read_csv(comparison_csv_path)
        
        st.markdown("### Detailed Metrics Table")
        st.dataframe(
            comparison_df.style.format({
                'Accuracy': '{:.4f}',
                'Precision': '{:.4f}',
                'Recall': '{:.4f}',
                'F1-Score': '{:.4f}',
                'Model Size (MB)': '{:.2f}'
            }),
            use_container_width=True
        )
    else:
        st.warning("❌ Model comparison CSV not found at: smartvision_dataset/results/model_comparison.csv")
    
    st.markdown("---")
    
    # ========================================================================
    # SECTION 3: ACCURACY METRICS VISUALIZATION
    # ========================================================================
    st.markdown("## 📈 Accuracy & Metrics Visualization")
    
    # Display pre-generated comparison chart
    comparison_chart_path = "smartvision_dataset/results/model_comparison_chart.png"
    if os.path.exists(comparison_chart_path):
        st.markdown("### Model Comparison Chart")
        st.image(comparison_chart_path, width=800, caption="Model Performance Comparison")
    
    # Create additional matplotlib visualizations
    if os.path.exists(comparison_csv_path):
        comparison_df = pd.read_csv(comparison_csv_path)
        
        viz_col1, viz_col2 = st.columns(2)
        
        # Accuracy Comparison
        with viz_col1:
            st.markdown("### Accuracy Comparison")
            fig, ax = plt.subplots(figsize=(8, 5))
            models = comparison_df['Model']
            accuracy = comparison_df['Accuracy'] * 100
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            ax.bar(models, accuracy, color=colors, edgecolor='black', linewidth=1.5)
            ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
            ax.set_xlabel('Model', fontsize=11, fontweight='bold')
            ax.set_ylim(0, 100)
            for i, v in enumerate(accuracy):
                ax.text(i, v + 2, f'{v:.2f}%', ha='center', fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        
        # Precision Comparison
        with viz_col2:
            st.markdown("### Precision Comparison")
            fig, ax = plt.subplots(figsize=(8, 5))
            precision = comparison_df['Precision'] * 100
            ax.bar(models, precision, color=colors, edgecolor='black', linewidth=1.5)
            ax.set_ylabel('Precision (%)', fontsize=11, fontweight='bold')
            ax.set_xlabel('Model', fontsize=11, fontweight='bold')
            ax.set_ylim(0, 100)
            for i, v in enumerate(precision):
                ax.text(i, v + 2, f'{v:.2f}%', ha='center', fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        
        viz_col3, viz_col4 = st.columns(2)
        
        # Recall Comparison
        with viz_col3:
            st.markdown("### Recall Comparison")
            fig, ax = plt.subplots(figsize=(8, 5))
            recall = comparison_df['Recall'] * 100
            ax.bar(models, recall, color=colors, edgecolor='black', linewidth=1.5)
            ax.set_ylabel('Recall (%)', fontsize=11, fontweight='bold')
            ax.set_xlabel('Model', fontsize=11, fontweight='bold')
            ax.set_ylim(0, 100)
            for i, v in enumerate(recall):
                ax.text(i, v + 2, f'{v:.2f}%', ha='center', fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        
        # F1-Score Comparison
        with viz_col4:
            st.markdown("### F1-Score Comparison")
            fig, ax = plt.subplots(figsize=(8, 5))
            f1_score = comparison_df['F1-Score'] * 100
            ax.bar(models, f1_score, color=colors, edgecolor='black', linewidth=1.5)
            ax.set_ylabel('F1-Score (%)', fontsize=11, fontweight='bold')
            ax.set_xlabel('Model', fontsize=11, fontweight='bold')
            ax.set_ylim(0, 100)
            for i, v in enumerate(f1_score):
                ax.text(i, v + 2, f'{v:.2f}%', ha='center', fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
    
    st.markdown("---")
    
    # ========================================================================
    # SECTION 4: INFERENCE SPEED COMPARISON
    # ========================================================================
    st.markdown("## ⚡ Inference Speed Comparison")
    st.info("⚠️ Estimated using model size (lighter models typically infer faster)")
    
    if os.path.exists(comparison_csv_path):
        comparison_df = pd.read_csv(comparison_csv_path)
        
        # Create efficiency visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        models = comparison_df['Model']
        sizes = comparison_df['Model Size (MB)']
        colors_efficiency = ['#2ca02c', '#d62728', '#ff7f0e', '#1f77b4']
        bars = ax.barh(models, sizes, color=colors_efficiency, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Model Size (MB)', fontsize=12, fontweight='bold')
        ax.set_title('Model Size Comparison (Lower = Faster Inference)', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        for i, v in enumerate(sizes):
            ax.text(v + 5, i, f'{v:.2f} MB', va='center', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    st.markdown("---")
    
    # ========================================================================
    # SECTION 5: CONFUSION MATRICES
    # ========================================================================
    st.markdown("## 🎯 Confusion Matrices")
    
    resnet_cm_path = "smartvision_dataset/results/confusion_matrix_ResNet50.png"
    yolo_cm_path = "smartvision_dataset/results/yolov8m_final_refinement/confusion_matrix.png"
    yolo_cm_norm_path = "smartvision_dataset/results/yolov8m_final_refinement/confusion_matrix_normalized.png"
    
    # Classification Confusion Matrix
    if os.path.exists(resnet_cm_path):
        st.markdown("### ResNet50 Classification Confusion Matrix")
        st.image(resnet_cm_path, use_column_width=True, caption="ResNet50 Confusion Matrix - All 26 Classes")
    else:
        st.warning(f"❌ ResNet50 confusion matrix not found at: {resnet_cm_path}")
    
    st.markdown("---")
    # ============================================================
    # YOLO CONFUSION MATRICES
    # ============================================================

    st.markdown(
        "### 🎯 YOLOv8m Object Detection Confusion Matrices"
    )

    # ------------------------------------------------------------
    # Standard Confusion Matrix
    # ------------------------------------------------------------

    if os.path.exists(yolo_cm_path):

        st.image(
            yolo_cm_path,
            use_column_width=True,
            caption="YOLOv8m Confusion Matrix"
    )

    else:

        st.warning(
        "❌ YOLO confusion matrix not found"
    )

    st.markdown("---")

    # ------------------------------------------------------------
    # Normalized Confusion Matrix
    # ------------------------------------------------------------

    if os.path.exists(yolo_cm_norm_path):

        st.image(
            yolo_cm_norm_path,
            use_column_width=True,
            caption="YOLOv8m Normalized Confusion Matrix"
    )
    else:
        st.warning(
            "❌ YOLO normalized confusion matrix not found"
    )

    st.markdown("---")
    
    # ========================================================================
    # SECTION 6: YOLO TRAINING PERFORMANCE
    # ========================================================================
    st.markdown("## 🚀 YOLOv8m Training Performance")
    
    yolo_results_path = "smartvision_dataset/results/yolov8m_final_refinement/results.png"
    yolo_boxf1_path = "smartvision_dataset/results/yolov8m_final_refinement/BoxF1_curve.png"
    yolo_boxpr_path = "smartvision_dataset/results/yolov8m_final_refinement/BoxPR_curve.png"
    
    # Training Results
    if os.path.exists(yolo_results_path):
        st.markdown("### Training Results Overview")
        st.image(yolo_results_path, use_column_width=True, caption="YOLOv8m Training Metrics")
    else:
        st.warning(f"❌ YOLO training results image not found")
    
    # Box F1 and PR Curves
    yolo_perf_col1, yolo_perf_col2 = st.columns(2)
    
    with yolo_perf_col1:
        if os.path.exists(yolo_boxf1_path):
            st.markdown("### Box F1 Curve")
            st.image(yolo_boxf1_path, use_column_width=True, caption="F1 Score vs Confidence Threshold")
        else:
            st.warning(f"❌ Box F1 curve not found")
    
    with yolo_perf_col2:
        if os.path.exists(yolo_boxpr_path):
            st.markdown("### Precision-Recall Curve")
            st.image(yolo_boxpr_path, use_column_width=True, caption="Precision vs Recall Trade-off")
        else:
            st.warning(f"❌ Precision-Recall curve not found")
    
    st.markdown("---")
    
    # ========================================================================
    # SECTION 7: CLASS-WISE PERFORMANCE BREAKDOWN
    # ========================================================================
    st.markdown("## 📋 Class-wise Performance Breakdown (ResNet50)")
    
    classification_report_path = "smartvision_dataset/results/classification_report_ResNet50.txt"
    
    if os.path.exists(classification_report_path):
        try:
            with open(classification_report_path, 'r') as f:
                report_content = f.read()
            st.code(report_content, language='text')
        except Exception as e:
            st.error(f"❌ Error reading classification report: {str(e)}")
    else:
        st.warning(f"❌ Classification report not found at: {classification_report_path}")
    
    st.markdown("---")
    
    # ========================================================================
    # SECTION 8: KEY INSIGHTS
    # ========================================================================
    st.markdown("## 💡 Key Insights")
    
    if os.path.exists(comparison_csv_path):
        comparison_df = pd.read_csv(comparison_csv_path)
        
        # Calculate insights
        best_model = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
        smallest_model = comparison_df.loc[comparison_df['Model Size (MB)'].idxmin()]
        best_f1 = comparison_df.loc[comparison_df['F1-Score'].idxmax()]
        best_precision = comparison_df.loc[comparison_df['Precision'].idxmax()]
        best_recall = comparison_df.loc[comparison_df['Recall'].idxmax()]
        
        insights_md = f"""
        📌 **Classification Performance:**
        - **{best_model['Model']}** achieved the highest classification accuracy of **{best_model['Accuracy'] * 100:.2f}%**
        - **{best_precision['Model']}** demonstrated the best precision score of **{best_precision['Precision'] * 100:.2f}%**
        - **{best_recall['Model']}** achieved the highest recall of **{best_recall['Recall'] * 100:.2f}%**
        - **{best_f1['Model']}** has the best F1-score balance of **{best_f1['F1-Score'] * 100:.2f}%**
        
        📦 **Model Efficiency:**
        - **{smallest_model['Model']}** is the most lightweight model at **{smallest_model['Model Size (MB)']:.2f} MB** - ideal for edge deployment
        - **{best_model['Model']}** provides the best accuracy-to-size trade-off for production deployment
        - MobileNetV2 and EfficientNetB0 are excellent choices for mobile and embedded systems
        
        🎯 **Detection Performance:**
        - **YOLOv8m** achieved reliable multi-object detection across 26 classes
        - The model shows consistent performance across multiple object classes
        - Confusion matrices indicate reliable detection and minimal false positives
        
        ⚡ **Recommendations:**
        - Use **{best_model['Model']}** when accuracy is the priority in resource-rich environments
        - Deploy **{smallest_model['Model']}** for lightweight, fast inference on edge devices
        - YOLOv8m is suitable for real-time multi-object detection applications
        """
        
        st.markdown(insights_md)
    
    st.markdown("---")
    
    # Summary
    st.markdown("""
    ### 📝 Summary
    SmartVision AI integrates multiple state-of-the-art deep learning models for comprehensive computer vision tasks:
    - **Classification Models:** ResNet50, VGG16, MobileNetV2, EfficientNetB0
    - **Detection Model:** YOLOv8m for real-time multi-object detection
    - **Performance:** Optimized for both accuracy and efficiency
    - **Deployment:** Suitable for production environments with flexible resource requirements
    """)

elif page == "ℹ️ About Project":

    # ========================================================================
    # ABOUT PROJECT PAGE
    # ========================================================================

    st.markdown("""
    <div class="page-header">
        <div class="page-title">
            ℹ️ About Project
        </div>
        <div class="page-subtitle">
          Learn about SmartVision AI and its capabilities
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '''
        <div class="info-box">
        <b>
        SmartVision AI - Intelligent Computer Vision
        using Deep Learning and Object Detection
        </b>
        </div>
        ''',
        unsafe_allow_html=True
    )

    # ============================================================
    # PROJECT OVERVIEW
    # ============================================================

    st.markdown("## 🚀 Project Overview")

    st.write("""
    **SmartVision AI** is an intelligent computer vision system
    developed for image classification and object detection
    using deep learning.

    The project combines multiple Convolutional Neural Network
    (CNN) architectures with **YOLOv8m** object detection to
    analyze images and identify objects with confidence scores
    and bounding boxes.

    SmartVision AI provides an interactive interface where users
    can upload images, classify objects, compare model
    performance, and visualize object detection results in
    real-time.
    """)

    st.markdown("---")

    # ============================================================
    # DATASET INFORMATION
    # ============================================================

    st.markdown("## 📂 Dataset Information")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 📊 Dataset Summary")

        st.info("""
        **Total Categories:** 26

        **Task Types:**
        - Image Classification
        - Object Detection

        **Image Processing:**
        - Resized to 224 × 224
        - Normalization
        - Data preprocessing
        - Training/Testing split
        """)

    with col2:

        st.markdown("### 🏷️ Object Categories")

        st.write("""
        Airplane, Bed, Bench, Bicycle, Bird,
        Bottle, Bowl, Bus, Cake, Car, Cat,
        Chair, Couch, Cow, Cup, Dog,
        Elephant, Horse, Motorcycle, Person,
        Pizza, Potted Plant, Stop Sign,
        Traffic Light, Train, Truck
        """)

    st.markdown("---")

    # ============================================================
    # MODEL ARCHITECTURES
    # ============================================================

    st.markdown("## 🧠 Model Architectures Used")

    model_col1, model_col2 = st.columns(2)

    with model_col1:

        st.markdown("""
        ### ResNet50
        - Best classification accuracy
        - Deep residual learning
        - Strong feature extraction
        """)

        st.markdown("""
        ### MobileNetV2
        - Lightweight architecture
        - Fast inference
        - Deployment-friendly
        """)

        st.markdown("""
        ### YOLOv8m
        - Real-time object detection
        - Multi-object recognition
        - Bounding box localization
        """)

    with model_col2:

        st.markdown("""
        ### VGG16
        - Deep CNN architecture
        - Feature learning
        - Strong baseline model
        """)

        st.markdown("""
        ### EfficientNetB0
        - Balanced efficiency
        - Optimized architecture
        - Good performance
        """)

    st.markdown("---")

    # ============================================================
    # TECH STACK
    # ============================================================

    st.markdown("## 🛠️ Technical Stack")

    tech_col1, tech_col2, tech_col3 = st.columns(3)

    with tech_col1:

        st.success("""
        **Programming**
        - Python
        """)

        st.success("""
        **Framework**
        - Streamlit
        """)

    with tech_col2:

        st.success("""
        **Deep Learning**
        - TensorFlow
        - Keras
        - YOLOv8
        """)

    with tech_col3:

        st.success("""
        **Libraries**
        - OpenCV
        - NumPy
        - Pandas
        - Matplotlib
        """)

    st.markdown("---")

    # ============================================================
    # PROJECT WORKFLOW
    # ============================================================

    st.markdown("## 🔄 Project Workflow")

    st.markdown("""
    ```text
    Image Upload
            ↓
    Image Preprocessing
            ↓
    Classification / Detection
            ↓
    Deep Learning Prediction
            ↓
    Result Visualization
    ```
    """)

    st.markdown("---")

    # ============================================================
    # KEY FEATURES
    # ============================================================

    st.markdown("## ✨ Key Features")

    feature_col1, feature_col2 = st.columns(2)

    with feature_col1:

        st.success("✅ Multi-model image classification")
        st.success("✅ YOLOv8m object detection")
        st.success("✅ Bounding box visualization")
        st.success("✅ Confidence threshold adjustment")

    with feature_col2:

        st.success("✅ Model comparison dashboard")
        st.success("✅ Performance analytics")
        st.success("✅ Confusion matrix visualization")
        st.success("✅ Interactive Streamlit interface")

    st.markdown("---")

    # ============================================================
    # PERFORMANCE SUMMARY
    # ============================================================

    st.markdown("## 📈 Performance Summary")

    perf_col1, perf_col2, perf_col3 = st.columns(3)

    with perf_col1:

        st.metric(
            label="🏆 Best Classification Model",
            value="ResNet50",
            delta="55.25% Accuracy"
        )

    with perf_col2:

        st.metric(
            label="⚡ Fastest Lightweight Model",
            value="MobileNetV2",
            delta="13.14 MB"
        )

    with perf_col3:

        st.metric(
            label="🎯 Detection Model",
            value="YOLOv8m",
            delta="Final Refinement"
        )

    st.markdown("---")

    # ============================================================
    # DEVELOPER INFORMATION
    # ============================================================

    st.markdown("## 👨‍💻 Developer Information")

    st.info("""
    **Developed By:** Rama Naren

    **Project Name:** SmartVision AI

    **Domain:** Computer Vision & Deep Learning

    **Technologies Used:**
    TensorFlow, YOLOv8m, Streamlit, OpenCV,
    Python, NumPy, Pandas, Matplotlib
    """)

    st.markdown("---")

    st.markdown(
        """
        <div style="text-align:center;">
            <h4>🤖 SmartVision AI</h4>
            <p>
            Built with Deep Learning, Computer Vision,
            and Streamlit
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )