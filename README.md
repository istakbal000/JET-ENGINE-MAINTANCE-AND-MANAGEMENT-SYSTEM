# 🚀 Jet Engine Predictive Maintenance System

## 📋 Project Overview

Advanced AI-powered predictive maintenance system for jet engines using 7 different ML approaches combined with Google Gemini AI for intelligent maintenance recommendations.

## 🎯 Features

### 🤖 Machine Learning Models
- **Exponential Degradation** - Mathematical exponential decay model
- **Similarity-Based** - Historical pattern matching
- **LSTM RUL** - Deep learning temporal analysis
- **LSTM Binary** - Healthy/Warning classification
- **RNN Multiclass** - Healthy/Warning/Critical classification
- **CNN Multiclass** - Spatial-temporal feature extraction
- **CNN-SVM Binary** - Hybrid CNN + SVM approach

### 🧠 AI Integration
- **Google Gemini 2.5 Flash** - Intelligent maintenance recommendations
- **Risk Assessment** - Automatic health evaluation
- **Professional Reports** - Structured maintenance guidance
- **Quick Recommendations** - Fast urgent advice

### 📊 Interactive Dashboard
- **Real-time Predictions** - Live RUL and health scores
- **Health Gauges** - Visual health indicators
- **Sensor Trends** - Time-series visualization
- **Model Comparison** - Performance comparison charts
- **Maintenance Alerts** - Automated warnings

## 🏗️ Project Structure

```
jet_engine_maintenance/
├── app_advanced.py           # Main Streamlit application
├── app_with_ai.py           # Simplified AI-only version
├── train_advanced.py          # Model training script
├── inspect_model.py          # Model inspection tool
├── requirements.txt          # Python dependencies
├── .env                    # Environment variables (API keys)
├── data/                   # NASA C-MAPSS dataset
│   ├── train_FD001.txt      # Training data
│   └── test_FD001.txt       # Test data
├── models/                  # Trained model files
│   └── lstm_rul_advanced.h5 # Main LSTM model
└── src/                     # Source code modules
    ├── preprocessing.py       # Data processing utilities
    ├── advanced_models.py     # 7 ML model implementations
    ├── utils.py              # Helper functions
    ├── gemini_advisor.py    # Original Gemini integration
    └── simple_gemini_advisor.py # Simplified AI advisor
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Conda/Anaconda recommended

### Installation Steps

1. **Clone Repository**
```bash
git clone <repository-url>
cd jet_engine_maintenance
```

2. **Create Environment**
```bash
conda create -n jet_engine python=3.10
conda activate jet_engine
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure API Key**
```bash
# Edit .env file
GEMINI_API_KEY=your_gemini_api_key_here
```

5. **Train Models**
```bash
python train_advanced.py
```

6. **Run Application**
```bash
streamlit run app_advanced.py
```

## 🚀 Quick Start

### Method 1: Full ML + AI (Recommended)
```bash
streamlit run app_advanced.py
```
- All 7 ML models
- Real predictions
- AI recommendations
- Complete dashboard

### Method 2: AI-Only Version
```bash
streamlit run app_with_ai.py
```
- Manual health input
- AI recommendations
- No TensorFlow conflicts
- Simplified interface

## 📊 Data Information

### NASA C-MAPSS Dataset
- **Source**: NASA Prognostics Data Repository
- **Engine Type**: Turbofan jet engines
- **Units**: 100 different engines
- **Sensors**: 21 sensor readings + 3 operational settings
- **Cycles**: 128-362 cycles per engine

### Data Features
- **Unit ID**: Engine identifier (1-100)
- **Cycle**: Time cycle number
- **Operational Settings**: Altitude, Mach number, throttle
- **Sensor Readings**: Temperature, pressure, vibration, etc.

## 🤖 Model Performance

### Regression Models (RUL Prediction)
| Model | MAE | MSE | Description |
|--------|------|------|-------------|
| Exponential Degradation | ~15-25 | ~400-600 | Mathematical decay model |
| Similarity-Based | ~12-20 | ~300-500 | Historical pattern matching |
| LSTM RUL | ~8-15 | ~200-400 | Deep learning temporal |

### Classification Models (Health Status)
| Model | Accuracy | Classes | Description |
|--------|----------|---------|-------------|
| LSTM Binary | ~85-90% | Healthy/Warning | Binary classification |
| RNN Multiclass | ~80-85% | 3-class health | Multi-class prediction |
| CNN Multiclass | ~82-88% | 3-class health | Spatial-temporal features |
| CNN-SVM Binary | ~86-91% | Healthy/Warning | Hybrid approach |

## 🎯 How to Use

### 1. Select Engine Unit
- Choose from available units (with ≥50 cycles)
- View unit information and cycle count

### 2. Choose Model
- Select from 7 different ML approaches
- Each model has different strengths

### 3. Analyze Results
- **Health Score**: 0-100% engine health
- **Predicted RUL**: Remaining useful life in cycles
- **Risk Level**: CRITICAL/HIGH/MODERATE/LOW

### 4. Get AI Recommendations
- Enable AI Advisor in sidebar
- Click "Get AI Maintenance Recommendations"
- Receive professional maintenance guidance

## 🔧 Configuration

### Environment Variables (.env)
```bash
# Google Gemini API Key
GEMINI_API_KEY=AIzaSyCGDm0_W5yI42vMqUZbg-1onirUl5bX5sE

# Additional variables (optional)
DATABASE_URL=your_database_url
OTHER_API_KEY=your_other_api_key
```

### Model Parameters
- **Sequence Length**: 50 cycles (configurable)
- **Health Thresholds**: 
  - Critical: <25% or <20 cycles
  - Warning: <50% or <50 cycles
- **API Temperature**: 0.7 (balanced creativity)

## 📈 API Endpoints

### Gemini AI Integration
- **Model**: gemini-2.5-flash
- **Endpoint**: https://generativelanguage.googleapis.com/v1beta
- **Method**: POST generateContent
- **Authentication**: API key in query parameter

### Request Format
```json
{
  "contents": [{
    "parts": [{"text": "maintenance prompt"}]
  }],
  "generationConfig": {
    "temperature": 0.7,
    "maxOutputTokens": 1024
  }
}
```

## 🚨 Maintenance Alerts

### Alert Levels
- **🔴 CRITICAL**: RUL < 20 cycles
- **🟡 WARNING**: RUL < 50 cycles  
- **🟢 NORMAL**: RUL ≥ 50 cycles

### Automated Actions
- **Immediate maintenance required** alerts
- **Schedule maintenance soon** warnings
- **Health gauge** visual indicators
- **Model consensus** analysis

## 📊 Model Comparison

### Strengths by Use Case

| Use Case | Best Model | Reason |
|-----------|-------------|---------|
| **Fast Prediction** | Exponential | Simple, quick calculation |
| **Pattern Recognition** | Similarity-Based | Historical matching |
| **Temporal Analysis** | LSTM RUL | Best for time series |
| **Binary Classification** | CNN-SVM | Hybrid approach accuracy |
| **Multi-class Health** | CNN Multiclass | Feature extraction |
| **Urgent Decisions** | LSTM Binary | Simple yes/no |

### Ensemble Approach
- **Weighted averaging** of multiple models
- **Confidence-based** model selection
- **Cross-validation** for reliability
- **AI consensus** analysis

## 🔍 Troubleshooting

### Common Issues

#### TensorFlow/Protobuf Conflicts
```bash
# Solution: Use app_with_ai.py (no TensorFlow)
streamlit run app_with_ai.py

# Or fix dependencies
pip install protobuf==4.25.0
pip install tensorflow
```

#### API Key Issues
```bash
# Check .env file exists
cat .env

# Verify API key format
# Should start with "AIza"
```

#### Model Not Found
```bash
# Train models first
python train_advanced.py

# Check models directory
ls models/
```

#### Data Loading Errors
```bash
# Verify data files exist
ls data/

# Check file permissions
chmod 644 data/*.txt
```

## 🎯 Performance Optimization

### Speed Improvements
- **Model caching** with `@st.cache_resource`
- **Data caching** with `@st.cache_data`
- **Lazy loading** of AI models
- **Batch processing** for multiple units

### Memory Management
- **Sequence length** optimization (50 cycles)
- **Feature selection** (18 sensors used)
- **Model pruning** for deployment
- **Data streaming** for large datasets

## 📚 Dependencies

### Core Libraries
```
numpy<2              # Numerical computing
pandas                # Data manipulation
scikit-learn          # ML utilities
tensorflow             # Deep learning
streamlit              # Web interface
matplotlib             # Plotting
plotly                # Interactive charts
google-generativeai     # AI integration
python-dotenv          # Environment variables
```

### Development Tools
```
pytest                 # Testing
black                  # Code formatting
jupyter               # Development
git                    # Version control
```

## 🚀 Deployment

### Local Development
```bash
# Run locally
streamlit run app_advanced.py

# Access interface
http://localhost:8501
```

### Production Deployment
```bash
# Streamlit Cloud
streamlit run app_advanced.py --server.port 80

# Docker deployment
docker build -t jet-engine-maintenance .
docker run -p 8501:8501 jet-engine-maintenance
```

### Environment Configuration
- **Development**: Local Streamlit
- **Staging**: Cloud deployment
- **Production**: Containerized with Docker

## 🔒 Security Considerations

### API Key Management
- **Environment variables** (not hardcoded)
- **Git ignore** `.env` file
- **Restricted access** to API endpoints
- **Rate limiting** for API calls

### Data Privacy
- **Local processing** of sensor data
- **No external storage** of predictions
- **Session-based** data handling
- **User consent** for AI features

## 📈 Future Enhancements

### Planned Features
- **Real-time data streaming** from live engines
- **Multi-engine fleet** management
- **Historical analysis** and trend prediction
- **Mobile app** for field technicians
- **Integration** with maintenance systems

### Model Improvements
- **Transformer architectures** for better temporal understanding
- **Graph neural networks** for component relationships
- **Ensemble methods** for improved accuracy
- **Transfer learning** from other engine types

## 📞 Support & Contributing

### Getting Help
- **Documentation**: This README file
- **Code comments**: Inline explanations
- **Error messages**: Descriptive troubleshooting
- **Model insights**: Performance analysis

### Contributing Guidelines
1. **Fork repository**
2. **Create feature branch**
3. **Write tests** for new features
4. **Follow code style** (PEP 8)
5. **Submit pull request**

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 🎉 Quick Summary

**What this project does:**
- Predicts jet engine remaining useful life
- Provides intelligent maintenance recommendations
- Uses 7 different ML approaches
- Integrates Google Gemini AI for expert advice

**Key benefits:**
- **Predictive maintenance** reduces downtime
- **AI-powered insights** improve decision making
- **Multi-model approach** ensures reliability
- **Interactive dashboard** for easy monitoring

**Perfect for:**
- **Airlines** maintenance teams
- **Aviation engineers** and technicians
- **Research** in predictive maintenance
- **Teaching** ML and AI applications

---

*🚀 Built with Streamlit, TensorFlow, and Google Gemini AI*
