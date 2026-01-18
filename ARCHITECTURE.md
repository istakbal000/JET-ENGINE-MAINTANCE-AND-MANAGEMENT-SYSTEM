# 🏗️ Technical Architecture Documentation

## 📋 System Overview

The Jet Engine Predictive Maintenance System is built on a modular architecture combining traditional machine learning with modern AI capabilities. The system processes NASA C-MAPSS turbofan engine data to predict remaining useful life (RUL) and provide intelligent maintenance recommendations.

## 🏛️ Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend Layer                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Streamlit     │  │   Plotly        │  │   Matplotlib │ │
│  │   Dashboard     │  │   Charts        │  │   Plots      │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Business Logic Layer                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Model         │  │   AI Advisor    │  │   Utils       │ │
│  │   Manager       │  │   (Gemini)      │  │   Module      │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Data          │  │   Model         │  │   Environment│ │
│  │   Processor     │  │   Storage       │  │   Variables  │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   External Services                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Gemini AI     │  │   File System   │  │   NASA       │ │
│  │   API           │  │   Storage       │  │   Dataset    │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Component Architecture

### 1. Frontend Layer (Streamlit)

**Purpose**: User interface and visualization
**Technologies**: Streamlit, Plotly, Matplotlib

**Key Components**:
- **Main Dashboard** (`app_advanced.py`)
- **Health Gauges** - Visual health indicators
- **Sensor Trends** - Time-series charts
- **Model Comparison** - Performance metrics
- **AI Recommendations** - Maintenance advice display

**Data Flow**:
```
User Input → Streamlit Widgets → Model Manager → Results Display
```

### 2. Business Logic Layer

#### Model Manager (`src/advanced_models.py`)
**Purpose**: Manages 7 different ML approaches
**Models**:
1. **ExponentialDegradationModel** - Mathematical decay
2. **SimilarityBasedModel** - Pattern matching
3. **LSTMRULModel** - Deep learning temporal
4. **LSTMBinaryModel** - Binary classification
5. **RNNMulticlassModel** - Multi-class classification
6. **CNN1DClassificationModel** - Spatial-temporal
7. **CNN1DSVMModel** - Hybrid CNN-SVM

**Architecture Pattern**: Strategy Pattern
```python
class ModelStrategy:
    def fit(self, X, y): pass
    def predict(self, X): pass
    def get_name(self): pass
```

#### AI Advisor (`src/simple_gemini_advisor.py`)
**Purpose**: Intelligent maintenance recommendations
**Technology**: Google Gemini 2.5 Flash API
**Architecture**: REST API Client

**Key Methods**:
- `generate_content()` - Generic AI generation
- `get_maintenance_recommendation()` - Detailed analysis
- `get_quick_recommendation()` - Fast advice

**Prompt Engineering**:
```python
prompt = f"""
You are an expert aviation maintenance engineer.
Analyze this jet engine health data:
- Health Score: {health_score:.1f}%
- Predicted RUL: {predicted_rul:.1f} cycles
- Risk Level: {risk_level}

PROVIDE:
1. Risk Assessment
2. Maintenance Recommendations
3. Inspection Points
4. Operational Guidelines
5. Cost Estimates
"""
```

### 3. Data Layer

#### Data Processor (`src/preprocessing.py`)
**Purpose**: Data loading, cleaning, and preprocessing
**Architecture**: Pipeline Pattern

**Processing Pipeline**:
1. **Data Loading** - NASA C-MAPSS dataset
2. **Feature Engineering** - RUL calculation
3. **Sensor Filtering** - Remove constant sensors
4. **Scaling** - MinMaxScaler normalization
5. **Sequence Generation** - Time series windows

**Key Methods**:
```python
def process_training_data(self, file_path, seq_length=50):
    # Load raw data
    # Calculate RUL
    # Filter sensors
    # Scale features
    # Generate sequences
    return X_train, y_train, train_df
```

#### Model Storage
**Purpose**: Persist trained models
**Format**: HDF5 for TensorFlow models
**Location**: `models/` directory

#### Environment Variables (`.env`)
**Purpose**: Secure configuration management
**Library**: python-dotenv

## 🔄 Data Flow Architecture

### 1. Training Pipeline
```
NASA Dataset → Data Processor → Feature Engineering → Model Training → Model Storage
```

### 2. Inference Pipeline
```
User Selection → Data Loading → Model Prediction → Health Analysis → AI Recommendations → UI Display
```

### 3. AI Integration Flow
```
Health Metrics → Prompt Generation → Gemini API → Response Processing → Display Recommendations
```

## 🗄️ Data Architecture

### NASA C-MAPSS Dataset Schema
```
train_FD001.txt / test_FD001.txt:
├── unit (int)           # Engine identifier (1-100)
├── cycle (int)          # Time cycle
├── op1, op2, op3 (float) # Operational settings
├── s1-s21 (float)       # Sensor readings
└── RUL (calculated)     # Remaining useful life
```

### Processed Data Schema
```
Processed Features:
├── unit_id
├── cycle_number
├── operational_settings (3 features)
├── sensor_readings (18 features after filtering)
├── rul_label
└── health_score (0-100%)
```

### Model Input Schema
```
Model Input Shape: (batch_size, sequence_length, n_features)
├── batch_size: Variable
├── sequence_length: 50 cycles
└── n_features: 21 (3 operational + 18 sensors)
```

## 🧠 Model Architecture Details

### 1. LSTM Models
```python
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(seq_length, n_features)),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')  # RUL prediction
])
```

### 2. CNN Models
```python
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(seq_length, n_features)),
    MaxPooling1D(pool_size=2),
    Conv1D(32, kernel_size=3, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')  # Multi-class classification
])
```

### 3. Hybrid CNN-SVM
```python
# CNN Feature Extractor
cnn_features = Conv1D(64, 3, activation='relu')(input_layer)
cnn_features = GlobalMaxPooling1D()(cnn_features)

# SVM Classifier
svm_output = SVC(kernel='rbf')(cnn_features)
```

## 🔌 API Architecture

### Gemini AI Integration
**Endpoint**: `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent`

**Request Structure**:
```python
{
    "contents": [{
        "parts": [{"text": "maintenance_prompt"}]
    }],
    "generationConfig": {
        "temperature": 0.7,
        "maxOutputTokens": 1024
    }
}
```

**Response Handling**:
```python
if response.status_code == 200:
    result = response.json()
    return result["candidates"][0]["content"]["parts"][0]["text"]
else:
    return f"API Error: {response.status_code}"
```

## 🔒 Security Architecture

### 1. API Key Management
- **Environment Variables**: Secure storage in `.env`
- **Git Ignore**: `.env` excluded from version control
- **Runtime Loading**: `load_dotenv()` at startup
- **Input Validation**: API key format validation

### 2. Data Privacy
- **Local Processing**: No external data transmission
- **Session Isolation**: Data isolated per user session
- **No Persistence**: No storage of user predictions
- **Secure Communication**: HTTPS for API calls

## ⚡ Performance Architecture

### 1. Caching Strategy
```python
@st.cache_resource
def load_models():
    # Cache loaded models across sessions

@st.cache_data
def load_and_process_data():
    # Cache processed data across sessions
```

### 2. Memory Management
- **Lazy Loading**: Models loaded on demand
- **Sequence Optimization**: Fixed 50-cycle windows
- **Feature Selection**: 18 sensors (reduced from 21)
- **Batch Processing**: Efficient tensor operations

### 3. Response Time Optimization
- **Parallel Processing**: Multiple model predictions
- **Async AI Calls**: Non-blocking API requests
- **Progress Indicators**: User feedback during processing
- **Error Handling**: Graceful degradation

## 🚀 Deployment Architecture

### 1. Development Environment
```
Local Machine:
├── Python 3.10 Environment
├── Conda Package Management
├── Jupyter Development
└── Streamlit Local Server (8501)
```

### 2. Production Options

#### Option A: Streamlit Cloud
```
Streamlit Cloud:
├── Automated Deployment
├── Git Integration
├── Shared Resources
└── Managed Infrastructure
```

#### Option B: Docker Container
```
Docker Container:
├── Python Base Image
├── Dependencies Pre-installed
├── Port Mapping (8501)
└── Volume Mounting for Data
```

#### Option C: Cloud VM
```
Cloud VM (AWS/GCP/Azure):
├── Full Control
├── Custom Configuration
├── Load Balancing
└── Auto-scaling Options
```

## 📊 Monitoring & Logging Architecture

### 1. Application Monitoring
- **Streamlit Metrics**: Built-in performance tracking
- **Model Performance**: Accuracy and latency tracking
- **API Usage**: Gemini API call monitoring
- **Error Tracking**: Exception logging and reporting

### 2. Model Monitoring
- **Prediction Accuracy**: Real-time model validation
- **Data Drift**: Input distribution monitoring
- **Model Drift**: Performance degradation detection
- **Confidence Scores**: Prediction reliability metrics

## 🔄 Scalability Architecture

### 1. Horizontal Scaling
- **Load Balancing**: Multiple Streamlit instances
- **Model Sharding**: Different models on different servers
- **Database Scaling**: Distributed data storage
- **API Rate Limiting**: Gemini API usage management

### 2. Vertical Scaling
- **GPU Acceleration**: TensorFlow GPU support
- **Memory Optimization**: Efficient data structures
- **CPU Optimization**: Multi-core processing
- **Storage Optimization**: SSD for faster I/O

## 🧪 Testing Architecture

### 1. Unit Testing
```python
# Test individual components
def test_data_processor():
    processor = DataProcessor()
    assert processor.calculate_rul(test_data) is not None

def test_model_prediction():
    model = LSTMRULModel()
    prediction = model.predict(test_input)
    assert prediction.shape == (1, 1)
```

### 2. Integration Testing
```python
# Test component interactions
def test_full_pipeline():
    # Data loading → Model prediction → AI recommendation
    assert full_pipeline_works()
```

### 3. Performance Testing
- **Load Testing**: Multiple concurrent users
- **Stress Testing**: Maximum capacity limits
- **Latency Testing**: Response time measurements
- **Memory Testing**: Resource usage monitoring

## 📈 Future Architecture Enhancements

### 1. Microservices Architecture
```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   Model     │  │   AI        │  │   Data      │
│   Service   │  │   Service   │  │   Service   │
└─────────────┘  └─────────────┘  └─────────────┘
       │                 │                 │
       └─────────────────┼─────────────────┘
                         │
              ┌─────────────────┐
              │   API Gateway   │
              └─────────────────┘
```

### 2. Real-time Processing
- **Apache Kafka**: Stream processing
- **Apache Spark**: Big data processing
- **Redis**: Real-time caching
- **WebSocket**: Live updates

### 3. Advanced AI Integration
- **Transformer Models**: BERT for text analysis
- **Graph Neural Networks**: Component relationships
- **Reinforcement Learning**: Maintenance scheduling
- **Multi-modal AI**: Text + sensor data fusion

---

## 🎯 Architecture Summary

This architecture provides:
- **Modularity**: Clear separation of concerns
- **Scalability**: Horizontal and vertical scaling options
- **Maintainability**: Well-structured code organization
- **Security**: Robust API key and data management
- **Performance**: Optimized caching and processing
- **Flexibility**: Easy to add new models and features

The system successfully combines traditional ML with modern AI to provide comprehensive jet engine predictive maintenance capabilities.
