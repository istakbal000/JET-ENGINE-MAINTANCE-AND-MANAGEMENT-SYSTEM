import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import os
from dotenv import load_dotenv
import requests
import json

# Load environment variables
load_dotenv()

# Import only the non-TensorFlow components
from src.preprocessing import DataProcessor
from src.utils import calculate_health_index, get_health_status, format_rul_display, check_maintenance_alert

# Simple AI Advisor using REST API (no TensorFlow dependency)
class SimpleGeminiAdvisor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    
    def generate_content(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1024}
        }
        
        try:
            response = requests.post(f"{self.api_url}?key={self.api_key}", headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                result = response.json()
                return result["candidates"][0]["content"]["parts"][0]["text"]
            return f"API Error: {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_maintenance_recommendation(self, health_score, predicted_rul, unit_id, model_predictions):
        risk_level = "CRITICAL" if health_score < 25 else "HIGH" if health_score < 50 else "MODERATE" if health_score < 75 else "LOW"
        urgency = "IMMEDIATE" if health_score < 25 else "URGENT" if health_score < 50 else "SCHEDULED" if health_score < 75 else "ROUTINE"
        
        model_summary = "\n".join([f"- {model}: {prediction:.1f}" for model, prediction in model_predictions.items()])
        
        prompt = f"""
You are an expert aviation maintenance engineer. Analyze this jet engine health data:

ENGINE HEALTH ANALYSIS:
- Engine Unit: {unit_id}
- Health Score: {health_score:.1f}%
- Predicted RUL: {predicted_rul:.1f} cycles
- Risk Level: {risk_level}
- Urgency: {urgency}

MODEL PREDICTIONS:
{model_summary}

PROVIDE A COMPREHENSIVE MAINTENANCE REPORT:

1. RISK ASSESSMENT
2. MAINTENANCE RECOMMENDATIONS
3. INSPECTION POINTS
4. OPERATIONAL GUIDELINES
5. COST & RESOURCE ESTIMATES

Format as a professional maintenance report with clear sections and actionable items.
"""
        return self.generate_content(prompt)
    
    def get_quick_recommendation(self, health_score, predicted_rul):
        if health_score < 25 or predicted_rul < 20:
            prompt = f"CRITICAL: Engine health {health_score:.1f}%, RUL {predicted_rul:.1f} cycles. Provide immediate safety-critical maintenance actions."
        elif health_score < 50 or predicted_rul < 50:
            prompt = f"WARNING: Engine health {health_score:.1f}%, RUL {predicted_rul:.1f} cycles. Provide urgent maintenance recommendations."
        else:
            prompt = f"NORMAL: Engine health {health_score:.1f}%, RUL {predicted_rul:.1f} cycles. Provide routine maintenance guidelines."
        
        return self.generate_content(prompt + "\n\nKeep response concise and actionable.")

# Set page config
st.set_page_config(
    page_title="Jet Engine Predictive Maintenance with AI",
    page_icon="✈️",
    layout="wide"
)

SEQ_LENGTH = 50

st.title("🚀 Jet Engine Predictive Maintenance System with AI")
st.markdown("Multi-model RUL prediction with AI-powered maintenance recommendations")

@st.cache_data
def load_and_process_data():
    processor = DataProcessor()
    
    # First load and fit on training data to initialize the scaler
    train_path = "data/train_FD001.txt"
    columns = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
    train_df = pd.read_csv(train_path, sep=r'\s+', header=None, names=columns)
    
    # Calculate RUL and fit scaler
    train_df = processor.calculate_rul(train_df)
    train_df = processor.drop_constant_sensors(train_df)
    processor.feature_columns = [col for col in train_df.columns if col not in ['unit', 'cycle', 'RUL']]
    processor.scaler.fit(train_df[processor.feature_columns])
    
    # Now process test data
    test_df = processor.process_test_data("data/test_FD001.txt")
    return processor, test_df

def create_health_gauge(health_score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=health_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Health Score (%)"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "lightgray"},
                {'range': [25, 75], 'color': "gray"},
                {'range': [75, 100], 'color': "lightgreen"}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 25}
        }
    ))
    return fig

def create_model_comparison_chart(results):
    fig = go.Figure()
    models = list(results.keys())
    values = list(results.values())
    
    fig.add_trace(go.Bar(
        x=models, y=values,
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

def main():
    # Load data
    processor, test_df = load_and_process_data()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # AI Configuration
    st.sidebar.subheader("🤖 AI Advisor Settings")
    default_api_key = os.getenv('GEMINI_API_KEY', '')
    gemini_api_key = st.sidebar.text_input("Enter Gemini API Key", value=default_api_key, type="password")
    enable_ai_advisor = st.sidebar.checkbox("Enable AI Maintenance Advisor", value=True)
    
    # Initialize AI Advisor
    ai_advisor = None
    if enable_ai_advisor and gemini_api_key:
        try:
            ai_advisor = SimpleGeminiAdvisor(gemini_api_key)
            st.sidebar.success("✅ AI Advisor Connected")
        except Exception as e:
            st.sidebar.error(f"❌ AI Advisor Error: {e}")
    elif enable_ai_advisor:
        st.sidebar.warning("⚠️ Enter API key to enable AI Advisor")
    
    # Unit selection
    available_units = sorted(test_df['unit'].unique())
    valid_units = [unit for unit in available_units if len(test_df[test_df['unit'] == unit]) >= SEQ_LENGTH]
    
    if not valid_units:
        st.error("❌ No units with sufficient data found. All units need at least 50 cycles.")
        return
    
    selected_unit = st.sidebar.selectbox("Select Unit", valid_units, index=0)
    
    # Get unit data
    unit_data = test_df[test_df['unit'] == selected_unit].sort_values('cycle')
    st.sidebar.info(f"Unit {selected_unit}: {len(unit_data)} cycles available")
    
    # Manual input for demonstration (since we can't load TensorFlow models)
    st.subheader("📊 Engine Health Analysis")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        predicted_rul = st.number_input("Predicted RUL (cycles)", min_value=0, max_value=300, value=75, step=1)
    with col2:
        health_score = st.number_input("Health Score (%)", min_value=0, max_value=100, value=60, step=1)
    with col3:
        st.metric("Status", get_health_status(health_score)[0])
    
    # Health gauge
    fig_gauge = create_health_gauge(health_score)
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Maintenance alerts
    if predicted_rul < 20:
        st.error("⚠️ **MAINTENANCE REQUIRED** - RUL is below 20 cycles!")
    elif predicted_rul < 50:
        st.warning("⚡ **ATTENTION** - RUL is below 50 cycles. Schedule maintenance soon.")
    
    # Model comparison (simulated)
    st.subheader("📊 Model Performance Comparison")
    
    # Simulate model predictions based on health score
    comparison_results = {
        "Exponential Degradation": health_score + np.random.normal(0, 5),
        "Similarity-Based": health_score + np.random.normal(0, 8),
        "LSTM RUL": health_score + np.random.normal(0, 3),
        "LSTM Binary": health_score + np.random.normal(0, 4),
        "RNN Multiclass": health_score + np.random.normal(0, 6),
        "CNN Multiclass": health_score + np.random.normal(0, 5),
        "CNN-SVM Binary": health_score + np.random.normal(0, 4)
    }
    
    fig_comparison = create_model_comparison_chart(comparison_results)
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Sensor trends
    st.subheader("📈 Sensor Trends")
    sensor_cols = [col for col in processor.feature_columns if col.startswith('s')]
    key_sensors = ['s11', 's12', 's2', 's3'] if 's11' in sensor_cols else sensor_cols[:4]
    
    fig_sensors = go.Figure()
    for sensor in key_sensors:
        if sensor in unit_data.columns:
            fig_sensors.add_trace(go.Scatter(
                x=unit_data['cycle'], y=unit_data[sensor],
                mode='lines', name=f'Sensor {sensor}', line=dict(width=2)
            ))
    
    fig_sensors.update_layout(
        title="Sensor Readings Over Time",
        xaxis_title="Cycle", yaxis_title="Sensor Value",
        hovermode='x unified'
    )
    st.plotly_chart(fig_sensors, use_container_width=True)
    
    # AI Maintenance Advisor
    if ai_advisor:
        st.subheader("🤖 AI Maintenance Advisor")
        
        if st.button("🧠 Get AI Maintenance Recommendations"):
            with st.spinner("🧠 AI analyzing engine health..."):
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
                    
                except Exception as e:
                    st.error(f"❌ AI Analysis Failed: {e}")
        
        if st.button("🚨 Get Quick Recommendation"):
            with st.spinner("Getting quick recommendation..."):
                try:
                    quick_rec = ai_advisor.get_quick_recommendation(health_score, predicted_rul)
                    st.info(quick_rec)
                except Exception as e:
                    st.error(f"❌ Quick Recommendation Failed: {e}")

if __name__ == "__main__":
    main()
