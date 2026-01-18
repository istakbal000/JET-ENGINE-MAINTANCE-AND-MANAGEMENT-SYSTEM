import google.generativeai as genai
import numpy as np
from typing import Dict, List, Optional
import json


class GeminiMaintenanceAdvisor:
    """AI-powered maintenance advisor using Google Gemini"""
    
    def __init__(self, api_key: str):
        """Initialize Gemini with API key"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
    def analyze_engine_health(self, 
                             unit_id: int,
                             predicted_rul: float,
                             health_score: float,
                             model_predictions: Dict[str, float],
                             sensor_trends: Optional[Dict] = None,
                             operational_context: Optional[Dict] = None) -> str:
        """
        Analyze engine health and provide maintenance recommendations
        
        Args:
            unit_id: Engine unit identifier
            predicted_rul: Predicted remaining useful life
            health_score: Health percentage (0-100)
            model_predictions: Dictionary of model predictions
            sensor_trends: Recent sensor data trends
            operational_context: Flight/operational conditions
            
        Returns:
            AI-generated maintenance recommendations
        """
        
        # Prepare context for Gemini
        context = self._prepare_analysis_context(
            unit_id, predicted_rul, health_score, 
            model_predictions, sensor_trends, operational_context
        )
        
        # Generate prompt
        prompt = self._create_maintenance_prompt(context)
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating AI recommendation: {str(e)}"
    
    def _prepare_analysis_context(self, 
                                 unit_id: int,
                                 predicted_rul: float,
                                 health_score: float,
                                 model_predictions: Dict[str, float],
                                 sensor_trends: Optional[Dict],
                                 operational_context: Optional[Dict]) -> Dict:
        """Prepare structured context for AI analysis"""
        
        # Determine risk level
        if health_score < 25:
            risk_level = "CRITICAL"
            urgency = "IMMEDIATE"
        elif health_score < 50:
            risk_level = "HIGH"
            urgency = "URGENT"
        elif health_score < 75:
            risk_level = "MODERATE"
            urgency = "SCHEDULED"
        else:
            risk_level = "LOW"
            urgency = "ROUTINE"
        
        # Analyze model consensus
        rul_predictions = {k: v for k, v in model_predictions.items() 
                           if isinstance(v, (int, float)) and v > 0}
        
        if rul_predictions:
            avg_rul = np.mean(list(rul_predictions.values()))
            rul_variance = np.var(list(rul_predictions.values()))
            model_consensus = "HIGH" if rul_variance < 100 else "LOW"
        else:
            avg_rul = predicted_rul
            rul_variance = 0
            model_consensus = "MEDIUM"
        
        return {
            "engine_unit": unit_id,
            "health_metrics": {
                "health_score_percent": health_score,
                "predicted_rul_cycles": predicted_rul,
                "average_rul_cycles": avg_rul,
                "risk_level": risk_level,
                "urgency": urgency
            },
            "model_analysis": {
                "individual_predictions": model_predictions,
                "consensus_level": model_consensus,
                "prediction_variance": rul_variance
            },
            "sensor_trends": sensor_trends or {},
            "operational_context": operational_context or {}
        }
    
    def _create_maintenance_prompt(self, context: Dict) -> str:
        """Create detailed prompt for Gemini analysis"""
        
        prompt = f"""
You are an expert aviation maintenance engineer with 20+ years of experience in jet engine predictive maintenance. 

ANALYZE THE FOLLOWING ENGINE HEALTH DATA:

ENGINE IDENTIFICATION:
- Unit ID: {context['engine_unit']}

HEALTH STATUS:
- Health Score: {context['health_metrics']['health_score_percent']:.1f}%
- Predicted RUL: {context['health_metrics']['predicted_rul_cycles']:.1f} cycles
- Risk Level: {context['health_metrics']['risk_level']}
- Urgency: {context['health_metrics']['urgency']}

MODEL PREDICTIONS ANALYSIS:
- Individual Model Results:
{self._format_model_predictions(context['model_analysis']['individual_predictions'])}
- Model Consensus: {context['model_analysis']['consensus_level']}
- Prediction Variance: {context['model_analysis']['prediction_variance']:.2f}

SENSOR TRENDS:
{self._format_sensor_trends(context['sensor_trends'])}

OPERATIONAL CONTEXT:
{self._format_operational_context(context['operational_context'])}

PROVIDE A COMPREHENSIVE MAINTENANCE RECOMMENDATION:

1. IMMEDIATE ACTIONS (if needed)
2. MAINTENANCE PRIORITY LEVEL
3. RECOMMENDED INSPECTION POINTS
4. SPECIFIC COMPONENTS TO CHECK
5. ESTIMATED MAINTENANCE TIMELINE
6. OPERATIONAL RESTRICTIONS (if any)
7. COST ESTIMATES (if possible)
8. ROOT CAUSE ANALYSIS HYPOTHESIS
9. PREVENTIVE MEASURES FOR FUTURE

Format your response as a professional maintenance report with clear sections and actionable recommendations. 
Be specific about what maintenance actions should be taken and when.
Consider safety as the highest priority.
"""
        
        return prompt
    
    def _format_model_predictions(self, predictions: Dict) -> str:
        """Format model predictions for prompt"""
        if not predictions:
            return "No model predictions available"
        
        formatted = []
        for model, value in predictions.items():
            if isinstance(value, (int, float)):
                formatted.append(f"  - {model}: {value:.2f}")
            else:
                formatted.append(f"  - {model}: {value}")
        
        return "\n".join(formatted)
    
    def _format_sensor_trends(self, sensor_trends: Dict) -> str:
        """Format sensor trends for prompt"""
        if not sensor_trends:
            return "No sensor trend data available"
        
        formatted = []
        for sensor, trend in sensor_trends.items():
            formatted.append(f"  - {sensor}: {trend}")
        
        return "\n".join(formatted)
    
    def _format_operational_context(self, context: Dict) -> str:
        """Format operational context for prompt"""
        if not context:
            return "No operational context available"
        
        formatted = []
        for key, value in context.items():
            formatted.append(f"  - {key}: {value}")
        
        return "\n".join(formatted)
    
    def get_quick_recommendation(self, health_score: float, predicted_rul: float) -> str:
        """Get quick maintenance recommendation based on key metrics"""
        
        if health_score < 25 or predicted_rul < 20:
            prompt = f"""
            Jet Engine Health Alert:
            - Health Score: {health_score:.1f}%
            - Predicted RUL: {predicted_rul:.1f} cycles
            
            Provide immediate, critical maintenance recommendations. Focus on safety and urgent actions needed.
            """
        elif health_score < 50 or predicted_rul < 50:
            prompt = f"""
            Jet Engine Health Warning:
            - Health Score: {health_score:.1f}%
            - Predicted RUL: {predicted_rul:.1f} cycles
            
            Provide maintenance recommendations for scheduling within the next 50 cycles.
            """
        else:
            prompt = f"""
            Jet Engine Health Status:
            - Health Score: {health_score:.1f}%
            - Predicted RUL: {predicted_rul:.1f} cycles
            
            Provide routine maintenance recommendations and monitoring guidelines.
            """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating recommendation: {str(e)}"
