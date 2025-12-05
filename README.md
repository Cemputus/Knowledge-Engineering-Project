# Smart Agriculture Advisor 🌾

A comprehensive **Hybrid Intelligent System** that combines **Knowledge Engineering (KE)** and **Deep Learning (DL)** for automated cassava disease diagnosis and agricultural advisory services.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Streamlit Web Interface](#streamlit-web-interface)
  - [Jupyter Notebook](#jupyter-notebook)
  - [Command Line Interface](#command-line-interface)
- [Technologies Used](#technologies-used)
- [Dataset Information](#dataset-information)
- [Model Performance](#model-performance)
- [Knowledge Engineering Components](#knowledge-engineering-components)
- [Deep Learning Components](#deep-learning-components)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

The **Smart Agriculture Advisor** is an intelligent system designed to help farmers and agricultural experts diagnose cassava diseases through:

1. **Image-based Diagnosis**: Using deep learning models to analyze leaf images
2. **Symptom-based Diagnosis**: Using expert rules and knowledge engineering for diagnosis based on observed symptoms
3. **Hybrid Integration**: Combining both approaches for more accurate and explainable results

### Problem Statement

Cassava is a critical food security crop in many regions, but it's susceptible to various diseases:
- **CBB** (Cassava Bacterial Blight)
- **CBSD** (Cassava Brown Streak Disease)
- **CGM** (Cassava Green Mottle)
- **CMD** (Cassava Mosaic Disease)
- **Healthy** plants

Early and accurate diagnosis is crucial for effective disease management and yield protection.

### Solution Approach

This project implements a **hybrid intelligent system** that:
- Uses **Deep Learning** (CNNs) for visual pattern recognition in leaf images
- Uses **Knowledge Engineering** (expert rules, ontologies) for symptom-based reasoning
- Combines both approaches for robust, explainable diagnosis

---

## ✨ Key Features

### 🖼️ Image-Based Diagnosis
- **Multiple Deep Learning Models**: Baseline CNN, MobileNetV2, EfficientNetB0
- **Real-time Prediction**: Upload images and get instant disease classification
- **Confidence Scores**: See prediction probabilities for all disease classes
- **Visualization**: View preprocessed images and model predictions

### 🔍 Symptom-Based Diagnosis
- **Expert Rule System**: 25+ expert rules for disease diagnosis
- **Interactive Symptom Selection**: Choose symptoms from a comprehensive list
- **Treatment Recommendations**: Get specific treatment advice based on diagnosis
- **Multi-crop Support**: Works with Maize, Rice, Wheat, Tomato, Potato, Soybean

### 📊 Statistics & Analytics
- **Dataset Statistics**: View class distribution, image counts, and data quality metrics
- **Model Performance**: Compare accuracy, precision, recall, and F1-scores
- **Visual Analytics**: Color distribution analysis, confusion matrices, training history

### 🎨 Modern User Interface
- **Streamlit Web App**: Beautiful, responsive, and interactive
- **Multi-page Navigation**: Easy access to all features
- **Real-time Updates**: Instant feedback on predictions and recommendations
- **Mobile-friendly**: Responsive design works on all devices

### 🔧 Advanced Features
- **Data Augmentation**: Multiple augmentation strategies for improved model generalization
- **Image Quality Validation**: Automatic checks for corrupted or invalid images
- **Class Imbalance Handling**: Weighted loss functions and balanced sampling
- **Transfer Learning**: Pre-trained models fine-tuned for cassava disease detection

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Smart Agriculture Advisor                   │
│                    Hybrid Intelligent System                  │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
    ┌───────────▼──────────┐    ┌───────────▼──────────┐
    │  Knowledge           │    │  Deep Learning      │
    │  Engineering (KE)    │    │  (DL)               │
    │                      │    │                     │
    │  • Ontology          │    │  • CNN Models       │
    │  • Expert Rules      │    │  • Transfer Learning│
    │  • Reasoning Engine  │    │  • Image Processing│
    │  • Rule Inference    │    │  • Data Augmentation│
    └───────────┬──────────┘    └───────────┬──────────┘
                │                           │
                └─────────────┬─────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Hybrid Fusion    │
                    │  & Integration    │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │  User Interface   │
                    │  (Streamlit)      │
                    └───────────────────┘
```

### Architecture Components

1. **Knowledge Engineering Layer**
   - OWL/RDF Ontology for agricultural concepts
   - Expert rules encoded in JSON format
   - Forward chaining inference engine
   - Treatment recommendation system

2. **Deep Learning Layer**
   - Convolutional Neural Networks (CNNs)
   - Transfer learning with MobileNetV2 and EfficientNetB0
   - Data preprocessing and augmentation pipeline
   - Model training and evaluation framework

3. **Integration Layer**
   - Combines DL predictions with KE reasoning
   - Confidence score fusion
   - Explainable AI through rule-based justifications

4. **Presentation Layer**
   - Streamlit web interface
   - Jupyter notebook for development
   - Command-line interface for batch processing

---

## 🚀 Installation

### Prerequisites

- **Python 3.8+** (Python 3.10+ recommended)
- **Git** for cloning the repository
- **8GB+ RAM** (16GB recommended for training)
- **GPU** (optional, but recommended for faster training)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Cemputus/Knowledge-Engineering-Project.git
cd Knowledge-Engineering-Project
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: If you encounter issues with TensorFlow, you may need to install it separately:

```bash
pip install tensorflow>=2.13.0
```

### Step 4: Verify Installation

```bash
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import streamlit as st; print('Streamlit installed')"
```

### Step 5: Download Dataset (Optional)

The dataset should be placed in the `cassava-disease/` directory with the following structure:

```
cassava-disease/
├── train/
│   ├── cbb/
│   ├── cbsd/
│   ├── cgm/
│   ├── cmd/
│   └── healthy/
├── test/
└── extraimages/
```

**Note**: Large files (>24MB) including model files (`.h5`) are excluded from git. You'll need to train the models or download pre-trained models separately.

---

## 📁 Project Structure

```
Knowledge-Engineering-Project/
│
├── 📄 README.md                          # This file
├── 📄 requirements.txt                   # Python dependencies
├── 📄 .gitignore                         # Git ignore rules
│
├── 📓 SmartAgricultureAdvisor.ipynb       # Main Jupyter notebook
├── 📓 KnowledgeEngineering.ipynb         # Knowledge engineering notebook
│
├── 🖥️ streamlit_ui.py                    # Streamlit web application
│
├── 📊 model_performance_summary.csv       # Model evaluation results
│
├── 🧠 Models/                            # Trained model files (not in git)
│   ├── baseline_cnn_model.h5
│   ├── mobilenet_cassava_model.h5
│   ├── efficientnet_cassava_model.h5
│   └── best_model.h5
│
├── 📚 ontology/                          # Knowledge Engineering components
│   ├── reasoner.py                       # Main reasoning engine
│   ├── agri_ontology.rdf                 # OWL/RDF ontology
│   ├── rules.swrl                        # SWRL rules
│   ├── test_examples.py                  # Test cases
│   └── requirements.txt                  # Ontology-specific dependencies
│
├── 📋 expert_rules.json                   # Expert rules for diagnosis
├── 🌾 agriculture_ontology.rdf           # Root ontology file
│
├── 📁 cassava-disease/                   # Dataset directory
│   ├── train/                            # Training images
│   │   ├── cbb/                          # Cassava Bacterial Blight
│   │   ├── cbsd/                         # Cassava Brown Streak Disease
│   │   ├── cgm/                          # Cassava Green Mottle
│   │   ├── cmd/                          # Cassava Mosaic Disease
│   │   └── healthy/                      # Healthy leaves
│   ├── test/                             # Test images
│   └── extraimages/                      # Additional unlabeled images
│
└── 📄 2025_UCU-CSE-EXAMS(...).pdf        # Project requirements document
```

---

## 💻 Usage

### Streamlit Web Interface

The easiest way to use the system is through the Streamlit web interface:

```bash
streamlit run streamlit_ui.py
```

This will start a local web server (usually at `http://localhost:8501`). The interface includes:

#### **Pages Available:**

1. **🏠 Home**
   - System overview
   - Quick start guide
   - Feature highlights

2. **🖼️ Image Diagnosis**
   - Upload cassava leaf images
   - Get instant disease predictions
   - View confidence scores
   - See treatment recommendations

3. **🔍 Symptom Diagnosis**
   - Select crop type
   - Choose observed symptoms
   - Get expert-based diagnosis
   - Receive treatment advice

4. **📊 Statistics**
   - Dataset statistics
   - Class distribution
   - Model performance metrics
   - Visual analytics

5. **📋 Expert Rules**
   - Browse all expert rules
   - Understand rule logic
   - View confidence scores

6. **ℹ️ About**
   - Project information
   - System architecture
   - Technology stack

### Jupyter Notebook

For development, experimentation, and training:

1. **Start Jupyter:**
   ```bash
   jupyter notebook
   ```

2. **Open `SmartAgricultureAdvisor.ipynb`**

3. **Follow the execution order:**
   - Section 1-3: Setup and imports
   - Section 4: Data exploration
   - Section 5: Knowledge engineering setup
   - Section 6: Deep learning setup and training
   - Section 7: Hybrid integration
   - Section 8: Evaluation
   - Section 9: Case studies

**Important**: Execute cells in order as some sections depend on previous ones.

### Command Line Interface

#### Using the Knowledge Engineering Reasoner

**Interactive Mode:**
```bash
cd ontology
python reasoner.py --interactive
```

**Command Line Mode:**
```bash
python reasoner.py --crop Maize --symptoms "YellowStreaks,StuntedGrowth"
```

**Example Diagnoses:**
```bash
# Maize disease
python reasoner.py --crop Maize --symptoms "YellowStreaks,StuntedGrowth"
# Output: Maize Streak Virus

# Tomato disease
python reasoner.py --crop Tomato --symptoms "WhitePowderyGrowth"
# Output: Powdery Mildew

# Rice disease
python reasoner.py --crop Rice --symptoms "LeafBlight"
# Output: Rice Blast
```

#### Testing the Reasoner

```bash
cd ontology
python test_examples.py
```

---

## 🛠️ Technologies Used

### Deep Learning
- **TensorFlow 2.13+**: Deep learning framework
- **Keras**: High-level neural network API
- **MobileNetV2**: Lightweight CNN architecture
- **EfficientNetB0**: Efficient CNN architecture

### Knowledge Engineering
- **OWL/RDF**: Ontology representation
- **SWRL**: Semantic Web Rule Language
- **Python**: Rule-based reasoning implementation

### Data Processing
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **Pillow (PIL)**: Image processing
- **scikit-learn**: Machine learning utilities

### Visualization
- **Matplotlib**: Plotting and visualization
- **Seaborn**: Statistical visualizations
- **Streamlit**: Web interface framework

### Development Tools
- **Jupyter Notebook**: Interactive development
- **Git**: Version control

---

## 📊 Dataset Information

### Cassava Disease Dataset

The dataset contains images of cassava leaves classified into 5 categories:

| Class | Full Name | Abbreviation | Description |
|-------|-----------|--------------|-------------|
| 0 | Cassava Bacterial Blight | CBB | Bacterial infection causing water-soaked lesions |
| 1 | Cassava Brown Streak Disease | CBSD | Viral disease with brown streaks on stems |
| 2 | Cassava Green Mottle | CGM | Viral disease with green mottling patterns |
| 3 | Cassava Mosaic Disease | CMD | Viral disease causing mosaic patterns |
| 4 | Healthy | Healthy | No disease detected |

### Dataset Statistics

- **Training Images**: ~21,000+ images
- **Test Images**: ~5,000+ images
- **Image Format**: JPEG/PNG
- **Image Size**: Variable (resized to 224x224 for models)
- **Class Distribution**: Imbalanced (requires class weighting)

### Data Preprocessing

1. **Image Quality Validation**
   - File integrity checks
   - Dimension validation
   - Color mode conversion (RGB)
   - File size validation

2. **Image Preprocessing**
   - Resize to 224x224 pixels
   - Normalization (0-1 range)
   - RGB conversion
   - Optional standardization

3. **Data Augmentation**
   - Random horizontal/vertical flips
   - Random brightness adjustment
   - Random contrast adjustment
   - Random saturation adjustment
   - Random rotation
   - Random cropping

---

## 📈 Model Performance

### Trained Models

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Baseline CNN** | 49.56% | 49.74% | 49.56% | 37.94% |
| **MobileNetV2** | **72.44%** | **72.22%** | **72.44%** | **71.61%** |
| EfficientNetB0 | 47.00% | 22.09% | 47.00% | 30.05% |

**Best Model**: MobileNetV2 (recommended for production use)

### Performance Notes

- **MobileNetV2** shows the best overall performance
- Models were trained with:
  - Class weights for imbalanced data
  - Data augmentation
  - Transfer learning (frozen base, then fine-tuning)
  - Early stopping and learning rate reduction

### Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

---

## 🧠 Knowledge Engineering Components

### Ontology Structure

The agricultural ontology (`agri_ontology.rdf`) defines:

- **Classes**: Crop, Disease, Pest, Symptom, Treatment
- **Properties**: hasSymptom, affectedBy, hasTreatment
- **Relationships**: Crop → Symptom → Disease → Treatment

### Expert Rules

The system includes **25+ expert rules** covering:

1. **Disease Diagnosis Rules** (CBB, CBSD, CGM, CMD)
   - Symptom-based identification
   - Confidence scoring
   - Multi-symptom combinations

2. **Treatment Rules**
   - Disease-specific treatments
   - Preventive measures
   - Integrated management strategies

3. **Severity Rules**
   - Infection rate assessment
   - Escalation procedures
   - Field management recommendations

### Reasoning Engine

The `AgricultureReasoner` class provides:

- **Forward Chaining**: Rule-based inference
- **Multi-crop Support**: Maize, Rice, Wheat, Tomato, Potato, Soybean
- **Symptom Matching**: Flexible symptom combination
- **Treatment Recommendations**: Actionable advice

### Example Rule

```json
{
  "id": 1,
  "name": "CBB_Rule_1",
  "conditions": ["Water_soaked_lesions", "Angular_leaf_spots"],
  "conclusion": "CBB",
  "confidence": 0.85,
  "description": "If water-soaked lesions AND angular leaf spots → Cassava Bacterial Blight"
}
```

---

## 🤖 Deep Learning Components

### Model Architectures

#### 1. Baseline CNN
- **Architecture**: Custom convolutional neural network
- **Layers**: Conv2D → MaxPooling → Dropout → Dense
- **Parameters**: ~2M trainable parameters
- **Use Case**: Baseline comparison

#### 2. MobileNetV2 (Recommended)
- **Architecture**: Transfer learning from ImageNet
- **Base Model**: MobileNetV2 (pre-trained)
- **Fine-tuning**: Two-stage training (frozen → unfrozen)
- **Advantages**: Lightweight, fast inference, good accuracy

#### 3. EfficientNetB0
- **Architecture**: EfficientNetB0 (pre-trained)
- **Advantages**: State-of-the-art efficiency
- **Use Case**: High-accuracy requirements

### Training Strategy

1. **Stage 1: Frozen Base**
   - Freeze pre-trained layers
   - Train only classification head
   - Learning rate: 0.001
   - Epochs: 10

2. **Stage 2: Fine-tuning**
   - Unfreeze top layers
   - Freeze first 100 layers
   - Lower learning rate: 1e-5
   - Epochs: 5

### Callbacks

- **Early Stopping**: Monitor validation loss, patience=5
- **ReduceLROnPlateau**: Reduce learning rate when stuck
- **ModelCheckpoint**: Save best model based on validation accuracy

### Class Imbalance Handling

- **Class Weights**: Automatically calculated using `sklearn.utils.class_weight`
- **Balanced Sampling**: Ensures fair representation during training

---

## 🔄 Hybrid Integration

The system combines DL and KE through:

1. **Image Prediction** (DL) → Disease class + confidence
2. **Symptom Analysis** (KE) → Disease diagnosis + treatment
3. **Fusion** → Combined confidence and recommendations

### Integration Benefits

- **Explainability**: KE provides rule-based justifications
- **Robustness**: Multiple approaches reduce errors
- **Flexibility**: Works with or without images
- **Accuracy**: Hybrid approach improves overall performance

---

## 🧪 Testing

### Test the Knowledge Engine

```bash
cd ontology
python test_examples.py
```

### Test the Streamlit UI

```bash
streamlit run streamlit_ui.py
# Navigate to http://localhost:8501
```

### Test Model Loading

```python
import tensorflow as tf
model = tf.keras.models.load_model('mobilenet_cassava_model.h5')
print("Model loaded successfully!")
```

---

## 📝 Development Workflow

### For Training New Models

1. **Prepare Data**: Ensure dataset is in `cassava-disease/train/`
2. **Open Notebook**: `SmartAgricultureAdvisor.ipynb`
3. **Run Sections 1-5**: Setup and data preparation
4. **Run Section 6**: Model training
5. **Evaluate**: Section 8 for performance metrics
6. **Save Model**: Models are automatically saved as `.h5` files

### For Adding New Rules

1. **Edit `expert_rules.json`**: Add new rule objects
2. **Update Reasoner**: Modify `ontology/reasoner.py` if needed
3. **Test**: Run `test_examples.py`
4. **Update UI**: Streamlit UI automatically reads from JSON

### For Extending to New Crops

1. **Update Ontology**: Add crop class to `agri_ontology.rdf`
2. **Add Rules**: Create rules in `expert_rules.json`
3. **Update Reasoner**: Add crop to knowledge base
4. **Test**: Verify with test cases

---

## 🐛 Troubleshooting

### Common Issues

#### 1. **TensorFlow Import Error**
```bash
pip install --upgrade tensorflow
```

#### 2. **CUDA/GPU Issues**
```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# If no GPU, TensorFlow will use CPU automatically
```

#### 3. **Model File Not Found**
- Models are excluded from git (too large)
- Train models using the notebook, or
- Download pre-trained models separately

#### 4. **Streamlit Port Already in Use**
```bash
streamlit run streamlit_ui.py --server.port 8502
```

#### 5. **Memory Errors During Training**
- Reduce batch size in notebook
- Use smaller image size
- Enable mixed precision training

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Contribution Guidelines

- Follow PEP 8 style guide
- Add docstrings to new functions
- Include tests for new features
- Update README if needed
- Keep commits atomic and descriptive

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Emmanuel Nsubuga** 
- **Lynn Amoit**
- **Edube Emmanuel Gulaba**
- **Tusiime Ronald**
- **Rugogamu Noela**
- **Anita Namaganda**
---

## 🙏 Acknowledgments

- **Dataset**: Cassava Disease Classification Dataset
- **TensorFlow Team**: For excellent deep learning framework
- **Streamlit Team**: For the amazing web framework
- **Open Source Community**: For various tools and libraries

---

## 📚 References

### Papers & Resources

1. **Transfer Learning**: "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" (Howard et al., 2017)
2. **EfficientNet**: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" (Tan & Le, 2019)
3. **Knowledge Engineering**: OWL 2 Web Ontology Language Primer

### Documentation

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [OWL 2 Specification](https://www.w3.org/TR/owl2-primer/)

---

## 🔮 Future Enhancements

- [ ] Support for more crop types
- [ ] Real-time video analysis
- [ ] Mobile app development
- [ ] Cloud deployment
- [ ] Multi-language support
- [ ] Advanced visualization features
- [ ] Integration with IoT sensors
- [ ] Yield prediction models
- [ ] Weather data integration
- [ ] Community features for farmers

---

## 📞 Contact

For questions, suggestions, or collaboration:

- **Email**: gulobaemmanueledube@gmail.com
- **GitHub**: [@Cemputus](https://github.com/Cemputus)
- **Project Repository**: [Knowledge-Engineering-Project](https://github.com/Cemputus/Knowledge-Engineering-Project)

---

## ⭐ Star History

If you find this project useful, please consider giving it a ⭐ on GitHub!

---

**Made with ❤️ for Agriculture and Food Security**

---

*Last Updated: December 2025*



# Smart Agriculture Advisor 🌾

A comprehensive **Hybrid Intelligent System** that combines **Knowledge Engineering (KE)** and **Deep Learning (DL)** for automated cassava disease diagnosis and agricultural advisory services.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Streamlit Web Interface](#streamlit-web-interface)
  - [Jupyter Notebook](#jupyter-notebook)
  - [Command Line Interface](#command-line-interface)
- [Technologies Used](#technologies-used)
- [Dataset Information](#dataset-information)
- [Model Performance](#model-performance)
- [Knowledge Engineering Components](#knowledge-engineering-components)
- [Deep Learning Components](#deep-learning-components)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

The **Smart Agriculture Advisor** is an intelligent system designed to help farmers and agricultural experts diagnose cassava diseases through:

1. **Image-based Diagnosis**: Using deep learning models to analyze leaf images
2. **Symptom-based Diagnosis**: Using expert rules and knowledge engineering for diagnosis based on observed symptoms
3. **Hybrid Integration**: Combining both approaches for more accurate and explainable results

### Problem Statement

Cassava is a critical food security crop in many regions, but it's susceptible to various diseases:
- **CBB** (Cassava Bacterial Blight)
- **CBSD** (Cassava Brown Streak Disease)
- **CGM** (Cassava Green Mottle)
- **CMD** (Cassava Mosaic Disease)
- **Healthy** plants

Early and accurate diagnosis is crucial for effective disease management and yield protection.

### Solution Approach

This project implements a **hybrid intelligent system** that:
- Uses **Deep Learning** (CNNs) for visual pattern recognition in leaf images
- Uses **Knowledge Engineering** (expert rules, ontologies) for symptom-based reasoning
- Combines both approaches for robust, explainable diagnosis

---

## ✨ Key Features

### 🖼️ Image-Based Diagnosis
- **Multiple Deep Learning Models**: Baseline CNN, MobileNetV2, EfficientNetB0
- **Real-time Prediction**: Upload images and get instant disease classification
- **Confidence Scores**: See prediction probabilities for all disease classes
- **Visualization**: View preprocessed images and model predictions

### 🔍 Symptom-Based Diagnosis
- **Expert Rule System**: 25+ expert rules for disease diagnosis
- **Interactive Symptom Selection**: Choose symptoms from a comprehensive list
- **Treatment Recommendations**: Get specific treatment advice based on diagnosis
- **Multi-crop Support**: Works with Maize, Rice, Wheat, Tomato, Potato, Soybean

### 📊 Statistics & Analytics
- **Dataset Statistics**: View class distribution, image counts, and data quality metrics
- **Model Performance**: Compare accuracy, precision, recall, and F1-scores
- **Visual Analytics**: Color distribution analysis, confusion matrices, training history

### 🎨 Modern User Interface
- **Streamlit Web App**: Beautiful, responsive, and interactive
- **Multi-page Navigation**: Easy access to all features
- **Real-time Updates**: Instant feedback on predictions and recommendations
- **Mobile-friendly**: Responsive design works on all devices

### 🔧 Advanced Features
- **Data Augmentation**: Multiple augmentation strategies for improved model generalization
- **Image Quality Validation**: Automatic checks for corrupted or invalid images
- **Class Imbalance Handling**: Weighted loss functions and balanced sampling
- **Transfer Learning**: Pre-trained models fine-tuned for cassava disease detection

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Smart Agriculture Advisor                   │
│                    Hybrid Intelligent System                  │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
    ┌───────────▼──────────┐    ┌───────────▼──────────┐
    │  Knowledge           │    │  Deep Learning      │
    │  Engineering (KE)    │    │  (DL)               │
    │                      │    │                     │
    │  • Ontology          │    │  • CNN Models       │
    │  • Expert Rules      │    │  • Transfer Learning│
    │  • Reasoning Engine  │    │  • Image Processing│
    │  • Rule Inference    │    │  • Data Augmentation│
    └───────────┬──────────┘    └───────────┬──────────┘
                │                           │
                └─────────────┬─────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Hybrid Fusion    │
                    │  & Integration    │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │  User Interface   │
                    │  (Streamlit)      │
                    └───────────────────┘
```

### Architecture Components

1. **Knowledge Engineering Layer**
   - OWL/RDF Ontology for agricultural concepts
   - Expert rules encoded in JSON format
   - Forward chaining inference engine
   - Treatment recommendation system

2. **Deep Learning Layer**
   - Convolutional Neural Networks (CNNs)
   - Transfer learning with MobileNetV2 and EfficientNetB0
   - Data preprocessing and augmentation pipeline
   - Model training and evaluation framework

3. **Integration Layer**
   - Combines DL predictions with KE reasoning
   - Confidence score fusion
   - Explainable AI through rule-based justifications

4. **Presentation Layer**
   - Streamlit web interface
   - Jupyter notebook for development
   - Command-line interface for batch processing

---

## 🚀 Installation

### Prerequisites

- **Python 3.8+** (Python 3.10+ recommended)
- **Git** for cloning the repository
- **8GB+ RAM** (16GB recommended for training)
- **GPU** (optional, but recommended for faster training)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Cemputus/Knowledge-Engineering-Project.git
cd Knowledge-Engineering-Project
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: If you encounter issues with TensorFlow, you may need to install it separately:

```bash
pip install tensorflow>=2.13.0
```

### Step 4: Verify Installation

```bash
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import streamlit as st; print('Streamlit installed')"
```

### Step 5: Download Dataset (Optional)

The dataset should be placed in the `cassava-disease/` directory with the following structure:

```
cassava-disease/
├── train/
│   ├── cbb/
│   ├── cbsd/
│   ├── cgm/
│   ├── cmd/
│   └── healthy/
├── test/
└── extraimages/
```

**Note**: Large files (>24MB) including model files (`.h5`) are excluded from git. You'll need to train the models or download pre-trained models separately.

---

## 📁 Project Structure

```
Knowledge-Engineering-Project/
│
├── 📄 README.md                          # This file
├── 📄 requirements.txt                   # Python dependencies
├── 📄 .gitignore                         # Git ignore rules
│
├── 📓 SmartAgricultureAdvisor.ipynb       # Main Jupyter notebook
├── 📓 KnowledgeEngineering.ipynb         # Knowledge engineering notebook
│
├── 🖥️ streamlit_ui.py                    # Streamlit web application
│
├── 📊 model_performance_summary.csv       # Model evaluation results
│
├── 🧠 Models/                            # Trained model files (not in git)
│   ├── baseline_cnn_model.h5
│   ├── mobilenet_cassava_model.h5
│   ├── efficientnet_cassava_model.h5
│   └── best_model.h5
│
├── 📚 ontology/                          # Knowledge Engineering components
│   ├── reasoner.py                       # Main reasoning engine
│   ├── agri_ontology.rdf                 # OWL/RDF ontology
│   ├── rules.swrl                        # SWRL rules
│   ├── test_examples.py                  # Test cases
│   └── requirements.txt                  # Ontology-specific dependencies
│
├── 📋 expert_rules.json                   # Expert rules for diagnosis
├── 🌾 agriculture_ontology.rdf           # Root ontology file
│
├── 📁 cassava-disease/                   # Dataset directory
│   ├── train/                            # Training images
│   │   ├── cbb/                          # Cassava Bacterial Blight
│   │   ├── cbsd/                         # Cassava Brown Streak Disease
│   │   ├── cgm/                          # Cassava Green Mottle
│   │   ├── cmd/                          # Cassava Mosaic Disease
│   │   └── healthy/                      # Healthy leaves
│   ├── test/                             # Test images
│   └── extraimages/                      # Additional unlabeled images
│
└── 📄 2025_UCU-CSE-EXAMS(...).pdf        # Project requirements document
```

---

## 💻 Usage

### Streamlit Web Interface

The easiest way to use the system is through the Streamlit web interface:

```bash
streamlit run streamlit_ui.py
```

This will start a local web server (usually at `http://localhost:8501`). The interface includes:

#### **Pages Available:**

1. **🏠 Home**
   - System overview
   - Quick start guide
   - Feature highlights

2. **🖼️ Image Diagnosis**
   - Upload cassava leaf images
   - Get instant disease predictions
   - View confidence scores
   - See treatment recommendations

3. **🔍 Symptom Diagnosis**
   - Select crop type
   - Choose observed symptoms
   - Get expert-based diagnosis
   - Receive treatment advice

4. **📊 Statistics**
   - Dataset statistics
   - Class distribution
   - Model performance metrics
   - Visual analytics

5. **📋 Expert Rules**
   - Browse all expert rules
   - Understand rule logic
   - View confidence scores

6. **ℹ️ About**
   - Project information
   - System architecture
   - Technology stack

### Jupyter Notebook

For development, experimentation, and training:

1. **Start Jupyter:**
   ```bash
   jupyter notebook
   ```

2. **Open `SmartAgricultureAdvisor.ipynb`**

3. **Follow the execution order:**
   - Section 1-3: Setup and imports
   - Section 4: Data exploration
   - Section 5: Knowledge engineering setup
   - Section 6: Deep learning setup and training
   - Section 7: Hybrid integration
   - Section 8: Evaluation
   - Section 9: Case studies

**Important**: Execute cells in order as some sections depend on previous ones.

### Command Line Interface

#### Using the Knowledge Engineering Reasoner

**Interactive Mode:**
```bash
cd ontology
python reasoner.py --interactive
```

**Command Line Mode:**
```bash
python reasoner.py --crop Maize --symptoms "YellowStreaks,StuntedGrowth"
```

**Example Diagnoses:**
```bash
# Maize disease
python reasoner.py --crop Maize --symptoms "YellowStreaks,StuntedGrowth"
# Output: Maize Streak Virus

# Tomato disease
python reasoner.py --crop Tomato --symptoms "WhitePowderyGrowth"
# Output: Powdery Mildew

# Rice disease
python reasoner.py --crop Rice --symptoms "LeafBlight"
# Output: Rice Blast
```

#### Testing the Reasoner

```bash
cd ontology
python test_examples.py
```

---

## 🛠️ Technologies Used

### Deep Learning
- **TensorFlow 2.13+**: Deep learning framework
- **Keras**: High-level neural network API
- **MobileNetV2**: Lightweight CNN architecture
- **EfficientNetB0**: Efficient CNN architecture

### Knowledge Engineering
- **OWL/RDF**: Ontology representation
- **SWRL**: Semantic Web Rule Language
