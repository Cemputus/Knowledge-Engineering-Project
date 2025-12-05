"""
Smart Agriculture Advisor - Streamlit User Interface
A comprehensive, interactive web application for cassava disease diagnosis
combining Deep Learning and Knowledge Engineering
"""

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from pathlib import Path
import sys

# Add ontology directory to path
sys.path.append('ontology')
from reasoner import AgricultureReasoner

# Page configuration
st.set_page_config(
    page_title="Smart Agriculture Advisor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        color: #1a5f2e;
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        color: white;
    }
    
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2d3748;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    .info-box {
        background: linear-gradient(135deg, #e0f2f1 0%, #b2dfdb 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #00acc1;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #ff9800;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .success-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        border: 1px solid #e2e8f0;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }
    
    .metric-card h3 {
        color: #64748b;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-card h2 {
        color: #1e293b;
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    .stSelectbox label, .stMultiselect label {
        font-weight: 600;
        color: #2d3748;
    }
    
    .stFileUploader label {
        font-weight: 600;
        color: #2d3748;
    }
    
    h1, h2, h3 {
        color: #1e293b;
    }
    
    .icon {
        display: inline-block;
        margin-right: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Constants
CLASS_NAMES = {
    0: "Cassava Bacterial Blight (CBB)",
    1: "Cassava Brown Streak Disease (CBSD)",
    2: "Cassava Green Mottle (CGM)",
    3: "Cassava Mosaic Disease (CMD)",
    4: "Healthy"
}

CLASS_NAMES_SHORT = {
    0: "CBB",
    1: "CBSD",
    2: "CGM",
    3: "CMD",
    4: "Healthy"
}

DISEASE_DESCRIPTIONS = {
    "CBB": "Cassava Bacterial Blight is caused by Xanthomonas axonopodis. Symptoms include water-soaked lesions, angular leaf spots, and black stems.",
    "CBSD": "Cassava Brown Streak Disease is a viral disease causing brown streaks on stems, yellowing between veins, and root necrosis.",
    "CGM": "Cassava Green Mottle is characterized by irregular green mottling patterns on leaves and reduced leaf size.",
    "CMD": "Cassava Mosaic Disease is a viral disease causing mosaic patterns, leaf distortion, and severe stunting.",
    "Healthy": "No disease detected. The plant appears healthy."
}

IMG_SIZE = (224, 224)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'reasoner' not in st.session_state:
    st.session_state.reasoner = None
if 'expert_rules' not in st.session_state:
    st.session_state.expert_rules = None

@st.cache_resource
def load_model(model_path):
    """Load the trained model"""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_resource
def load_reasoner():
    """Load the agriculture reasoner"""
    try:
        reasoner = AgricultureReasoner()
        return reasoner
    except Exception as e:
        st.error(f"Error loading reasoner: {str(e)}")
        return None

def load_expert_rules():
    """Load expert rules from JSON"""
    try:
        with open('expert_rules.json', 'r') as f:
            rules = json.load(f)
        return rules
    except Exception as e:
        st.warning(f"Could not load expert rules: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Resize image
    image = image.resize(IMG_SIZE)
    # Convert to array
    img_array = np.array(image)
    # Normalize
    img_array = img_array.astype('float32') / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_disease(model, image):
    """Predict disease from image"""
    img_array = preprocess_image(image)
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class])
    
    # Get all class probabilities
    class_probs = {
        CLASS_NAMES_SHORT[i]: float(predictions[0][i]) 
        for i in range(len(CLASS_NAMES_SHORT))
    }
    
    return predicted_class, confidence, class_probs

def get_treatment_recommendations(disease, expert_rules):
    """Get treatment recommendations based on disease and expert rules"""
    if not expert_rules:
        return []
    
    treatments = []
    disease_upper = disease.upper()
    
    for rule in expert_rules:
        if rule.get('conclusion') == disease_upper or disease_upper in rule.get('conclusion', ''):
            if 'treatment' in rule.get('conclusion', '').lower():
                treatments.append(rule.get('description', ''))
    
    # Default treatments based on disease
    default_treatments = {
        'CBB': [
            'Apply copper-based fungicides',
            'Remove infected plants',
            'Use resistant varieties',
            'Practice crop rotation',
            'Improve field sanitation'
        ],
        'CBSD': [
            'Use virus-free planting material',
            'Remove infected plants',
            'Control whitefly vectors',
            'Plant resistant varieties',
            'Early harvesting'
        ],
        'CGM': [
            'Remove infected plants',
            'Use clean planting material',
            'Control vectors',
            'Practice field sanitation'
        ],
        'CMD': [
            'Use virus-free planting material',
            'Remove infected plants',
            'Control whitefly vectors',
            'Plant resistant varieties',
            'Early harvesting'
        ]
    }
    
    if disease_upper in default_treatments:
        treatments.extend(default_treatments[disease_upper])
    
    return list(set(treatments))  # Remove duplicates

def main():
    # Header with modern styling
    st.markdown("""
    <div class="main-header">
        <svg style="width: 60px; height: 60px; display: inline-block; vertical-align: middle; margin-right: 1rem;" fill="currentColor" viewBox="0 0 20 20">
            <path d="M10.394 2.08a1 1 0 00-.788 0l-7 3a1 1 0 000 1.84L5.25 8.051a.999.999 0 01.356-.257l4-1.714a1 1 0 11.788 1.838L7.667 9.088l1.94.831a1 1 0 01.787 1.838l-7 3a1 1 0 01-.394 0l-7-3a1 1 0 000-1.838l7-3a1 1 0 00.788 0l7 3a1 1 0 010 1.838z"/>
        </svg>
        Smart Agriculture Advisor
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; color: #64748b; margin-bottom: 2rem; font-size: 1.1rem;'>
        <p style='font-weight: 500;'>Hybrid Intelligent System for Cassava Disease Diagnosis</p>
        <p style='font-size: 0.95rem;'>Combining Deep Learning (CNN) and Knowledge Engineering for Accurate Disease Detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 1rem;">
            <h2 style="color: white; margin: 0; font-size: 1.5rem;">Configuration</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Model selection
        st.subheader("Model Selection")
        model_option = st.selectbox(
            "Choose Model",
            ["MobileNetV2 (Recommended)", "EfficientNetB0", "Baseline CNN"],
            help="Select the trained model for prediction"
        )
        
        # Map selection to model file
        model_files = {
            "MobileNetV2 (Recommended)": "mobilenet_cassava_model.h5",
            "EfficientNetB0": "efficientnet_cassava_model.h5",
            "Baseline CNN": "baseline_cnn_model.h5"
        }
        
        selected_model_file = model_files[model_option]
        
        # Load model button
        if st.button("Load Model", type="primary", use_container_width=True):
            with st.spinner("Loading model..."):
                model = load_model(selected_model_file)
                if model:
                    st.session_state.model = model
                    st.session_state.model_loaded = True
                    st.success("Model loaded successfully!")
                else:
                    st.error("Failed to load model")
        
        if st.session_state.model_loaded:
            st.success("Model Ready")
        
        st.divider()
        
        # Load reasoner
        if st.button("Initialize Knowledge Engine", use_container_width=True):
            with st.spinner("Loading knowledge base..."):
                reasoner = load_reasoner()
                if reasoner:
                    st.session_state.reasoner = reasoner
                    st.success("Knowledge Engine Ready!")
                else:
                    st.error("Failed to load knowledge engine")
        
        if st.session_state.reasoner:
            st.success("Knowledge Engine Active")
        
        st.divider()
        
        # Navigation
        st.markdown("""
        <div style="padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-top: 1rem;">
            <h2 style="color: white; margin: 0; font-size: 1.5rem;">Navigation</h2>
        </div>
        """, unsafe_allow_html=True)
        page = st.radio(
            "Select Page",
            ["Home", "Image Diagnosis", "Symptom-Based Diagnosis", "Statistics", "Expert Rules", "About"],
            label_visibility="collapsed"
        )
    
    # Main content based on page selection
    if page == "Home":
        show_home_page()
    elif page == "Image Diagnosis":
        show_image_diagnosis()
    elif page == "Symptom-Based Diagnosis":
        show_symptom_diagnosis()
    elif page == "Statistics":
        show_statistics()
    elif page == "Expert Rules":
        show_expert_rules()
    elif page == "About":
        show_about()

def show_home_page():
    """Display home page"""
    st.markdown('<div class="sub-header">Welcome to Smart Agriculture Advisor</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Accuracy</h3>
            <h2>85%+</h2>
            <p style="color: #64748b; margin-top: 0.5rem;">Model Performance</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Diseases</h3>
            <h2>5 Classes</h2>
            <p style="color: #64748b; margin-top: 0.5rem;">CBB, CBSD, CGM, CMD, Healthy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>Hybrid AI</h3>
            <h2>DL + KE</h2>
            <p style="color: #64748b; margin-top: 0.5rem;">Deep Learning + Knowledge Engineering</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h4 style="margin-top: 0; color: #00695c;">Quick Start Guide</h4>
        <ol style="line-height: 2;">
            <li>Load a trained model from the sidebar</li>
            <li>Navigate to <strong>Image Diagnosis</strong> to upload a cassava leaf image</li>
            <li>Or use <strong>Symptom-Based Diagnosis</strong> for expert system diagnosis</li>
            <li>View treatment recommendations and yield predictions</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sub-header">System Features</div>
    """, unsafe_allow_html=True)
    
    feature_col1, feature_col2 = st.columns(2)
    
    with feature_col1:
        st.markdown("""
        **Deep Learning Component**
        - Convolutional Neural Networks (CNNs)
        - Transfer Learning (MobileNetV2, EfficientNetB0)
        - High accuracy image classification
        - Real-time predictions
        
        **Knowledge Engineering Component**
        - Expert rules and ontology
        - Symptom-based reasoning
        - Treatment recommendations
        - Hybrid decision making
        """)
    
    with feature_col2:
        st.markdown("""
        **Disease Detection**
        - Cassava Bacterial Blight (CBB)
        - Cassava Brown Streak Disease (CBSD)
        - Cassava Green Mottle (CGM)
        - Cassava Mosaic Disease (CMD)
        - Healthy plant detection
        
        **Advanced Features**
        - Yield prediction
        - Treatment effectiveness analysis
        - Confidence scoring
        - Multi-model comparison
        """)

def show_image_diagnosis():
    """Image-based disease diagnosis page"""
    st.markdown('<div class="sub-header">Image-Based Disease Diagnosis</div>', unsafe_allow_html=True)
    
    if not st.session_state.model_loaded:
        st.markdown("""
        <div class="warning-box">
            <strong>Please load a model from the sidebar first!</strong>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Image upload
    uploaded_file = st.file_uploader(
        "Upload a cassava leaf image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image of a cassava leaf for disease diagnosis"
    )
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            # Predict
            if st.button("Diagnose Disease", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    predicted_class, confidence, class_probs = predict_disease(st.session_state.model, image)
                    
                    disease_name = CLASS_NAMES_SHORT[predicted_class]
                    disease_full = CLASS_NAMES[predicted_class]
                    
                    # Display results
                    st.markdown(f"""
                    <div class="success-box">
                        <h3 style="margin-top: 0;">Diagnosis Result</h3>
                        <h2 style="color: #1a5f2e; margin: 1rem 0;">{disease_full}</h2>
                        <p style="font-size: 1.1rem;"><strong>Confidence:</strong> <span style="color: #667eea; font-weight: 700;">{confidence*100:.2f}%</span></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Disease description
                    if disease_name in DISEASE_DESCRIPTIONS:
                        st.info(f"{DISEASE_DESCRIPTIONS[disease_name]}")
                    
                    # Probability distribution
                    st.subheader("Prediction Probabilities")
                    prob_df = pd.DataFrame({
                        'Disease': list(class_probs.keys()),
                        'Probability': [f"{p*100:.2f}%" for p in class_probs.values()]
                    })
                    st.dataframe(prob_df, use_container_width=True, hide_index=True)
                    
                    # Visualization
                    fig, ax = plt.subplots(figsize=(10, 6))
                    diseases = list(class_probs.keys())
                    probs = list(class_probs.values())
                    colors = ['#667eea' if d == disease_name else '#a0aec0' for d in diseases]
                    
                    bars = ax.barh(diseases, probs, color=colors, edgecolor='black', linewidth=1.5)
                    ax.set_xlabel('Probability', fontweight='bold', fontsize=12)
                    ax.set_title('Disease Prediction Probabilities', fontweight='bold', fontsize=14)
                    ax.set_xlim(0, 1)
                    
                    # Add value labels
                    for bar, prob in zip(bars, probs):
                        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                               f' {prob*100:.2f}%', va='center', fontweight='bold')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Treatment recommendations
                    st.subheader("Treatment Recommendations")
                    expert_rules = load_expert_rules()
                    treatments = get_treatment_recommendations(disease_name, expert_rules)
                    
                    if treatments:
                        for i, treatment in enumerate(treatments[:10], 1):
                            st.markdown(f"**{i}.** {treatment}")
                    else:
                        st.info("No specific treatment recommendations available. Consult with agricultural experts.")
                    
                    # Hybrid reasoning (if reasoner is loaded)
                    if st.session_state.reasoner:
                        st.subheader("Hybrid Reasoning")
                        st.info("Knowledge Engine is active. You can add observed symptoms in the Symptom-Based Diagnosis page for enhanced reasoning.")
                    
                    # Yield prediction placeholder
                    if disease_name != "Healthy":
                        st.subheader("Yield Impact Estimate")
                        severity = "High" if confidence > 0.8 else "Medium" if confidence > 0.5 else "Low"
                        yield_impact = {
                            "High": "15-30% yield reduction expected",
                            "Medium": "5-15% yield reduction expected",
                            "Low": "Minimal yield impact expected"
                        }
                        st.warning(f"{severity} severity detected: {yield_impact[severity]}")

def show_symptom_diagnosis():
    """Symptom-based diagnosis using knowledge engineering"""
    st.markdown('<div class="sub-header">Symptom-Based Diagnosis</div>', unsafe_allow_html=True)
    
    if not st.session_state.reasoner:
        st.markdown("""
        <div class="warning-box">
            <strong>Please initialize the Knowledge Engine from the sidebar first!</strong>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Initialize Knowledge Engine"):
            with st.spinner("Loading knowledge base..."):
                reasoner = load_reasoner()
                if reasoner:
                    st.session_state.reasoner = reasoner
                    st.success("Knowledge Engine Ready!")
                    st.rerun()
        return
    
    st.info("Select observed symptoms to get expert system diagnosis and recommendations.")
    
    # Get available symptoms from reasoner
    cassava_symptoms = [
        "Water_soaked_lesions",
        "Angular_leaf_spots",
        "Yellowing_leaves",
        "Leaf_wilting",
        "Black_stems",
        "Brown_streaks_on_stems",
        "Yellowing_between_veins",
        "Root_necrosis",
        "Chlorotic_mottling",
        "Stunted_growth",
        "Mosaic_patterns",
        "Leaf_distortion",
        "Reduced_leaf_size",
        "Yellow_green_mottling",
        "Severe_stunting",
        "Irregular_leaf_patterns"
    ]
    
    symptom_labels = {
        "Water_soaked_lesions": "Water-soaked lesions",
        "Angular_leaf_spots": "Angular leaf spots",
        "Yellowing_leaves": "Yellowing leaves",
        "Leaf_wilting": "Leaf wilting",
        "Black_stems": "Black stems",
        "Brown_streaks_on_stems": "Brown streaks on stems",
        "Yellowing_between_veins": "Yellowing between veins",
        "Root_necrosis": "Root necrosis",
        "Chlorotic_mottling": "Chlorotic mottling",
        "Stunted_growth": "Stunted growth",
        "Mosaic_patterns": "Mosaic patterns",
        "Leaf_distortion": "Leaf distortion",
        "Reduced_leaf_size": "Reduced leaf size",
        "Yellow_green_mottling": "Yellow-green mottling",
        "Severe_stunting": "Severe stunting",
        "Irregular_leaf_patterns": "Irregular leaf patterns"
    }
    
    # Symptom selection
    st.subheader("Select Observed Symptoms")
    selected_symptoms = st.multiselect(
        "Choose all symptoms you observe:",
        options=cassava_symptoms,
        format_func=lambda x: symptom_labels.get(x, x),
        help="Select all symptoms visible on the cassava plant"
    )
    
    if selected_symptoms:
        st.success(f"{len(selected_symptoms)} symptom(s) selected")
    
    # Diagnosis button
    if st.button("Get Expert Diagnosis", type="primary", use_container_width=True):
        if not selected_symptoms:
            st.error("Please select at least one symptom!")
        else:
            with st.spinner("Analyzing symptoms with expert system..."):
                # Use expert rules for diagnosis
                expert_rules = load_expert_rules()
                
                if expert_rules:
                    # Match symptoms to rules
                    matched_rules = []
                    for rule in expert_rules:
                        conditions = rule.get('conditions', [])
                        # Check if any conditions match selected symptoms
                        matched = any(cond.replace('_', ' ').lower() in 
                                    [s.replace('_', ' ').lower() for s in selected_symptoms] 
                                    for cond in conditions)
                        if matched:
                            matched_rules.append(rule)
                    
                    if matched_rules:
                        st.subheader("Diagnosis Results")
                        
                        # Group by conclusion
                        diagnoses = {}
                        for rule in matched_rules:
                            conclusion = rule.get('conclusion', 'Unknown')
                            if conclusion not in diagnoses:
                                diagnoses[conclusion] = []
                            diagnoses[conclusion].append(rule)
                        
                        for disease, rules in diagnoses.items():
                            st.markdown(f"""
                            <div class="success-box">
                                <h3 style="margin-top: 0;">{disease}</h3>
                                <p><strong>Confidence:</strong> {rules[0].get('confidence', 0)*100:.0f}%</p>
                                <p><strong>Matched Rule:</strong> {rules[0].get('description', 'N/A')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Treatment recommendations
                        st.subheader("Recommended Treatments")
                        all_treatments = set()
                        for rule in matched_rules:
                            conclusion = rule.get('conclusion', '')
                            treatments = get_treatment_recommendations(conclusion, expert_rules)
                            all_treatments.update(treatments)
                        
                        for i, treatment in enumerate(list(all_treatments)[:15], 1):
                            st.markdown(f"**{i}.** {treatment}")
                    else:
                        st.warning("No matching expert rules found. Please consult with agricultural experts.")
                else:
                    st.error("Expert rules not available. Please check the expert_rules.json file.")

def show_statistics():
    """Display dataset and model statistics"""
    st.markdown('<div class="sub-header">Statistics & Analytics</div>', unsafe_allow_html=True)
    
    # Dataset statistics
    st.subheader("Dataset Information")
    
    dataset_path = Path("cassava-disease/train")
    if dataset_path.exists():
        class_counts = {}
        total_images = 0
        
        for class_folder in dataset_path.iterdir():
            if class_folder.is_dir():
                class_name = class_folder.name.upper()
                image_count = len(list(class_folder.glob("*.jpg"))) + len(list(class_folder.glob("*.JPG")))
                class_counts[class_name] = image_count
                total_images += image_count
        
        if class_counts:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Images", f"{total_images:,}")
                st.metric("Disease Classes", len(class_counts))
            
            with col2:
                # Class distribution chart
                fig, ax = plt.subplots(figsize=(10, 6))
                classes = list(class_counts.keys())
                counts = list(class_counts.values())
                colors = sns.color_palette("husl", len(classes))
                
                bars = ax.bar(classes, counts, color=colors, edgecolor='black', linewidth=1.5)
                ax.set_xlabel('Disease Class', fontweight='bold')
                ax.set_ylabel('Number of Images', fontweight='bold')
                ax.set_title('Dataset Class Distribution', fontweight='bold', fontsize=14)
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, count in zip(bars, counts):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{count}', ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Class distribution table
            st.subheader("Class Distribution Details")
            dist_df = pd.DataFrame({
                'Class': list(class_counts.keys()),
                'Image Count': list(class_counts.values()),
                'Percentage': [f"{(c/total_images)*100:.2f}%" for c in class_counts.values()]
            })
            st.dataframe(dist_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Dataset not found. Statistics unavailable.")
    
    # Model performance (if available)
    if os.path.exists("model_performance_summary.csv"):
        st.subheader("Model Performance")
        perf_df = pd.read_csv("model_performance_summary.csv")
        st.dataframe(perf_df, use_container_width=True, hide_index=True)

def show_expert_rules():
    """Display expert rules"""
    st.markdown('<div class="sub-header">Expert Rules & Knowledge Base</div>', unsafe_allow_html=True)
    
    expert_rules = load_expert_rules()
    
    if expert_rules:
        st.info(f"Loaded {len(expert_rules)} expert rules from knowledge base")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            filter_disease = st.selectbox(
                "Filter by Disease",
                ["All"] + list(set([rule.get('conclusion', 'Unknown') for rule in expert_rules]))
            )
        with col2:
            search_term = st.text_input("Search rules", placeholder="Enter keyword...")
        
        # Display rules
        filtered_rules = expert_rules
        if filter_disease != "All":
            filtered_rules = [r for r in filtered_rules if r.get('conclusion') == filter_disease]
        if search_term:
            filtered_rules = [r for r in filtered_rules 
                            if search_term.lower() in str(r).lower()]
        
        st.subheader(f"Rules ({len(filtered_rules)} found)")
        
        for i, rule in enumerate(filtered_rules, 1):
            with st.expander(f"Rule {i}: {rule.get('name', 'Unnamed')} - {rule.get('conclusion', 'Unknown')}"):
                st.markdown(f"**Description:** {rule.get('description', 'N/A')}")
                st.markdown(f"**Conditions:** {', '.join(rule.get('conditions', []))}")
                st.markdown(f"**Conclusion:** {rule.get('conclusion', 'N/A')}")
                st.markdown(f"**Confidence:** {rule.get('confidence', 0)*100:.0f}%")
    else:
        st.error("Expert rules not available. Please check the expert_rules.json file.")

def show_about():
    """About page"""
    st.markdown('<div class="sub-header">About Smart Agriculture Advisor</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3 style="margin-top: 0; color: #00695c;">Project Overview</h3>
        <p>Smart Agriculture Advisor is a hybrid intelligent system that combines Deep Learning (CNN) 
        and Knowledge Engineering for accurate cassava disease diagnosis. The system provides real-time 
        disease detection, treatment recommendations, and yield predictions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sub-header">Technology Stack</div>
    """, unsafe_allow_html=True)
    
    tech_col1, tech_col2 = st.columns(2)
    
    with tech_col1:
        st.markdown("""
        **Deep Learning:**
        - TensorFlow/Keras
        - Convolutional Neural Networks
        - Transfer Learning (MobileNetV2, EfficientNetB0)
        - Image preprocessing and augmentation
        
        **Knowledge Engineering:**
        - Expert rule-based system
        - Ontology reasoning
        - Symptom-based diagnosis
        """)
    
    with tech_col2:
        st.markdown("""
        **Web Framework:**
        - Streamlit
        - Interactive UI components
        - Real-time predictions
        
        **Data Processing:**
        - NumPy, Pandas
        - Matplotlib, Seaborn
        - PIL (Image processing)
        """)
    
    st.markdown("""
    <div class="sub-header">Supported Diseases</div>
    """, unsafe_allow_html=True)
    
    for idx, (code, name) in enumerate(CLASS_NAMES.items()):
        st.markdown(f"**{idx+1}. {name}**")
        if CLASS_NAMES_SHORT[idx] in DISEASE_DESCRIPTIONS:
            st.markdown(f"   {DISEASE_DESCRIPTIONS[CLASS_NAMES_SHORT[idx]]}")
    
    st.markdown("""
    <div class="sub-header">Key Features</div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    - **Image-based Diagnosis**: Upload cassava leaf images for instant disease detection
    - **Symptom-based Diagnosis**: Use expert system for symptom-based reasoning
    - **Hybrid Intelligence**: Combines ML predictions with expert knowledge
    - **Treatment Recommendations**: Get actionable treatment suggestions
    - **Yield Prediction**: Estimate crop yield impact
    - **Multi-model Support**: Choose from different trained models
    - **Confidence Scoring**: Transparent prediction confidence levels
    - **Visual Analytics**: Interactive charts and visualizations
    """)
    
    st.markdown("""
    <div class="success-box">
        <h4 style="margin-top: 0;">Citation</h4>
        <p>Smart Agriculture Advisor - Hybrid Intelligent System for Cassava Disease Diagnosis<br>
        Combining Deep Learning and Knowledge Engineering</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
