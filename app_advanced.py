import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import os
from dotenv import load_dotenv
from src.preprocessing import DataProcessor
from src.advanced_models import (
    ExponentialDegradationModel, SimilarityBasedModel, LSTMRULModel,
    LSTMClassificationModel, RNNClassificationModel, CNN1DClassificationModel,
    CNN1DSVMModel, convert_rul_to_classes
)
from src.utils import calculate_health_index, get_health_status, format_rul_display, check_maintenance_alert
from src.simple_gemini_advisor import SimpleGeminiAdvisor


# Set page config
st.set_page_config(
    page_title="Advanced Jet Engine Predictive Maintenance",
    page_icon="✈️",
    layout="wide"
)

# Sequence length (cycles) used for model inputs
SEQ_LENGTH = 50

# Title and description
st.title("🚀 Advanced Jet Engine Predictive Maintenance System")
st.markdown("Multi-model RUL prediction and health monitoring with 7 different ML approaches")

@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    
    try:
        # Load LSTM RUL model with custom objects
        models['lstm_rul'] = tf.keras.models.load_model(
            "models/lstm_rul_advanced.h5",
            custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
        )
        st.success("✅ LSTM RUL model loaded")
    except Exception as e:
        st.error(f"❌ LSTM RUL model not found or corrupted: {e}")
        st.info("💡 Please run train_advanced.py first to train the models.")
        return None
    
    # Initialize other models (they need to be trained on data)
    models['exponential'] = ExponentialDegradationModel()
    models['similarity'] = SimilarityBasedModel()
    models['lstm_binary'] = LSTMClassificationModel(n_classes=2)
    models['rnn_multi'] = RNNClassificationModel(n_classes=3)
    models['cnn_multi'] = CNN1DClassificationModel(n_classes=3)
    models['cnn_svm'] = CNN1DSVMModel()
    
    return models

@st.cache_data
def load_and_process_data():
    """Load and process training and test data"""
    processor = DataProcessor()
    
    # Load and process training data
    X_train, y_train, train_df = processor.process_training_data("data/train_FD001.txt", seq_length=SEQ_LENGTH)
    
    # Load and process test data
    test_df = processor.process_test_data("data/test_FD001.txt")
    
    return processor, X_train, y_train, train_df, test_df

def create_model_comparison_chart(results):
    """Create comparison chart for model performances"""
    fig = go.Figure()
    
    models = list(results.keys())
    values = list(results.values())
    
    fig.add_trace(go.Bar(
        x=models,
        y=values,
        text=[f"{v:.2f}" for v in values],
        textposition='auto',
        marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
    ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Models",
        yaxis_title="Performance Metric",
        showlegend=False
    )
    
    return fig

def create_health_gauge(health_score):
    """Create health gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = health_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Health Score (%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "lightgray"},
                {'range': [25, 75], 'color': "gray"},
                {'range': [75, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 25
            }
        }
    ))
    
    return fig

def main():
    # Load environment variables
    load_dotenv()

    # Load models and data
    models = load_models()
    if models is None:
        return
    
    processor, X_train, y_train, train_df, test_df = load_and_process_data()
    
    # Train simple models on the fly
    with st.spinner("Training models..."):
        models['exponential'].fit(X_train, y_train)
        models['similarity'].fit(X_train, y_train)
        
        # Convert labels for classification models
        y_train_binary = convert_rul_to_classes(y_train, n_classes=2)
        y_train_multi = convert_rul_to_classes(y_train, n_classes=3)
        
        # Convert to categorical for multiclass models
        y_train_multi_cat = to_categorical(y_train_multi, num_classes=3)
        
        models['lstm_binary'].fit(X_train, y_train_binary, epochs=5, validation_split=0.1)
        models['rnn_multi'].fit(X_train, y_train_multi_cat, epochs=5, validation_split=0.1)
        models['cnn_multi'].fit(X_train, y_train_multi_cat, epochs=5, validation_split=0.1)
        models['cnn_svm'].fit(X_train, y_train_binary, epochs=5)
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Gemini AI Configuration
    st.sidebar.subheader("🤖 AI Advisor Settings")
    
    # Try to get API key from environment first
    default_api_key = os.getenv('GEMINI_API_KEY', '')
    gemini_api_key = st.sidebar.text_input("Enter Gemini API Key", 
                                          value=default_api_key,
                                          type="password")
    enable_ai_advisor = st.sidebar.checkbox("Enable AI Maintenance Advisor", value=True)
    
    # Initialize Gemini Advisor
    ai_advisor = None
    if enable_ai_advisor and gemini_api_key:
        try:
            ai_advisor = SimpleGeminiAdvisor(gemini_api_key)
            st.sidebar.success("✅ AI Advisor Connected")
        except Exception as e:
            st.sidebar.error(f"❌ AI Advisor Error: {e}")
    elif enable_ai_advisor:
        st.sidebar.warning("⚠️ Enter API key to enable AI Advisor")
    
    # Model selection
    model_options = {
        "Exponential Degradation": "exponential",
        "Similarity-Based": "similarity", 
        "LSTM RUL": "lstm_rul",
        "LSTM Binary": "lstm_binary",
        "RNN Multiclass": "rnn_multi",
        "CNN Multiclass": "cnn_multi",
        "CNN-SVM Binary": "cnn_svm"
    }
    
    selected_model_name = st.sidebar.selectbox("Select Model", list(model_options.keys()))
    selected_model = model_options[selected_model_name]
    
    # Unit selection - use test data for consistency
    available_units = sorted(test_df['unit'].unique())
    
    # Filter units with enough cycles (>= SEQ_LENGTH) in test data
    valid_units = []
    for unit in available_units:
        unit_cycles = len(test_df[test_df['unit'] == unit])
        if unit_cycles >= SEQ_LENGTH:
            valid_units.append(unit)
    
    if not valid_units:
        st.error("❌ No units with sufficient data found. All units need at least 50 cycles.")
        return
    
    selected_unit = st.sidebar.selectbox("Select Unit", valid_units, index=0)
    
    # Show data info for selected unit
    unit_data = train_df[train_df['unit'] == selected_unit]
    unit_cycles = len(unit_data)
    st.sidebar.info(f"Unit {selected_unit}: {unit_cycles} cycles available")
    
    # Get data for selected unit
    unit_data = test_df[test_df['unit'] == selected_unit].sort_values('cycle')
    
    # Prepare data for prediction
    if len(unit_data) >= SEQ_LENGTH:
        # Get last SEQ_LENGTH cycles
        last_50_cycles = unit_data[processor.feature_columns].iloc[-SEQ_LENGTH:].values
        input_data = np.expand_dims(last_50_cycles, axis=0)
        
        # Make prediction
        if selected_model in ['lstm_binary', 'rnn_multi', 'cnn_multi', 'cnn_svm']:
            # Classification models
            prediction = models[selected_model].predict(input_data)
            
            if selected_model == 'lstm_binary' or selected_model == 'cnn_svm':
                # Binary classification
                prob_class_1 = prediction[0][1] if len(prediction.shape) > 1 else prediction[0]
                predicted_class = "Warning" if prob_class_1 > 0.5 else "Healthy"
                confidence = max(prob_class_1, 1 - prob_class_1) * 100
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Status", predicted_class)
                with col2:
                    st.metric("Confidence", f"{confidence:.1f}%")
                
                # Health gauge
                health_score = 100 - prob_class_1 * 100
                fig_gauge = create_health_gauge(health_score)
                st.plotly_chart(fig_gauge, use_container_width=True)
                
            else:
                # Multiclass classification
                predicted_class_idx = np.argmax(prediction[0])
                class_names = ["Healthy", "Warning", "Critical"]
                predicted_class = class_names[predicted_class_idx]
                confidence = prediction[0][predicted_class_idx] * 100
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Status", predicted_class)
                with col2:
                    st.metric("Confidence", f"{confidence:.1f}%")
                
                # Health gauge
                health_scores = [100, 50, 10]
                health_score = health_scores[predicted_class_idx]
                fig_gauge = create_health_gauge(health_score)
                st.plotly_chart(fig_gauge, use_container_width=True)
        
        else:
            # Regression models (RUL prediction)
            predicted_rul = models[selected_model].predict(input_data)[0]
            health_score = calculate_health_index(predicted_rul)
            health_status, status_color = get_health_status(health_score)
            
            # Main metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted RUL", format_rul_display(predicted_rul))
            with col2:
                st.metric("Health Score", f"{health_score:.1f}%")
            with col3:
                st.metric("Status", health_status)
            
            # Health gauge
            fig_gauge = create_health_gauge(health_score)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Maintenance alert
            if check_maintenance_alert(predicted_rul):
                st.error("⚠️ **MAINTENANCE REQUIRED** - RUL is below 20 cycles!")
            elif predicted_rul < 50:
                st.warning("⚡ **ATTENTION** - RUL is below 50 cycles. Schedule maintenance soon.")
        
        # Model comparison
        st.subheader("📊 Model Performance Comparison")
        
        # Test all models on current data
        comparison_results = {}
        for name, model_key in model_options.items():
            if model_key in ['lstm_binary', 'rnn_multi', 'cnn_multi', 'cnn_svm']:
                pred = models[model_key].predict(input_data)
                if model_key in ['lstm_binary', 'cnn_svm']:
                    if len(pred.shape) > 1 and pred.shape[1] > 1:
                        prob = pred[0][1]  # Warning probability
                    else:
                        prob = pred[0] if len(pred.shape) == 1 else pred[0][0]
                    comparison_results[name] = prob * 100  # Warning probability
                else:
                    pred_class = np.argmax(pred[0])
                    comparison_results[name] = pred_class * 33.33  # Convert to percentage
            else:
                pred_rul = models[model_key].predict(input_data)[0]
                comparison_results[name] = calculate_health_index(pred_rul)
        
        # Create comparison chart
        fig_comparison = create_model_comparison_chart(comparison_results)
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Sensor trends
        st.subheader("📈 Sensor Trends")
        
        # Plot key sensors
        sensor_cols = [col for col in processor.feature_columns if col.startswith('s')]
        key_sensors = ['s11', 's12', 's2', 's3'] if 's11' in sensor_cols else sensor_cols[:4]
        
        fig_sensors = go.Figure()
        
        for sensor in key_sensors:
            if sensor in unit_data.columns:
                fig_sensors.add_trace(go.Scatter(
                    x=unit_data['cycle'],
                    y=unit_data[sensor],
                    mode='lines',
                    name=f'Sensor {sensor}',
                    line=dict(width=2)
                ))
        
        fig_sensors.update_layout(
            title="Sensor Readings Over Time",
            xaxis_title="Cycle",
            yaxis_title="Sensor Value",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_sensors, use_container_width=True)
        
        # AI Maintenance Advisor
        if ai_advisor and (selected_model not in ['lstm_binary', 'rnn_multi', 'cnn_multi', 'cnn_svm']):
            st.subheader("🤖 AI Maintenance Advisor")
            
            with st.spinner("🧠 AI analyzing engine health..."):
                # Prepare sensor trends for AI
                sensor_trends = {}
                for sensor in key_sensors:
                    if sensor in unit_data.columns:
                        recent_values = unit_data[sensor].tail(10).values
                        trend = "increasing" if recent_values[-1] > recent_values[0] else "decreasing"
                        sensor_trends[sensor] = f"{trend} (current: {recent_values[-1]:.2f})"
                
                # Get AI recommendations
                try:
                    ai_recommendation = ai_advisor.get_maintenance_recommendation(
                        health_score=health_score,
                        predicted_rul=predicted_rul,
                        unit_id=selected_unit,
                        model_predictions=comparison_results
                    )
                    
                    st.success("✅ AI Analysis Complete")
                    st.markdown("### 📋 AI Maintenance Recommendations")
                    st.write(ai_recommendation)
                    
                    # Quick recommendation button
                    if st.button("🚨 Get Quick Recommendation"):
                        quick_rec = ai_advisor.get_quick_recommendation(health_score, predicted_rul)
                        st.info(quick_rec)
                        
                except Exception as e:
                    st.error(f"❌ AI Analysis Failed: {e}")
        
        # Model information
        st.subheader("ℹ️ Model Information")
        
        model_info = {
            "Exponential Degradation": "Mathematical model assuming exponential decay of engine health",
            "Similarity-Based": "Finds similar historical patterns and averages their RUL values",
            "LSTM RUL": "Deep learning model capturing long-term temporal dependencies",
            "LSTM Binary": "Classifies engine as Healthy/Warning based on sensor patterns",
            "RNN Multiclass": "Classifies engine into Healthy/Warning/Critical categories",
            "CNN Multiclass": "Uses convolutional layers to extract spatial-temporal features",
            "CNN-SVM Binary": "Hybrid model combining CNN feature extraction with SVM classification"
        }
        
        st.info(f"**{selected_model_name}**: {model_info[selected_model_name]}")
        
    else:
        st.error(f"❌ Insufficient data for prediction. Unit must have at least {SEQ_LENGTH} cycles of data.")


if __name__ == "__main__":
    main()
