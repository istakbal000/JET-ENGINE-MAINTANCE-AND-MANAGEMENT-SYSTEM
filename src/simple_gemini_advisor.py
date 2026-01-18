import requests
import json
from typing import Dict, Optional


class SimpleGeminiAdvisor:
    """Simplified Gemini AI advisor using REST API to avoid protobuf conflicts"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    
    def generate_content(self, prompt: str) -> str:
        """Generate content using Gemini REST API"""
        headers = {
            "Content-Type": "application/json",
        }
        
        data = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 1024,
            }
        }
        
        try:
            response = requests.post(
                f"{self.api_url}?key={self.api_key}",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                return f"API Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Connection Error: {str(e)}"
    
    def get_maintenance_recommendation(self, 
                                     health_score: float,
                                     predicted_rul: float,
                                     unit_id: int,
                                     model_predictions: Dict[str, float]) -> str:
        """Get maintenance recommendation based on engine health data"""
        
        # Determine risk level
        if health_score < 25:
            risk_level = "CRITICAL"
            urgency = "IMMEDIATE ATTENTION REQUIRED"
        elif health_score < 50:
            risk_level = "HIGH"
            urgency = "URGENT MAINTENANCE NEEDED"
        elif health_score < 75:
            risk_level = "MODERATE"
            urgency = "SCHEDULE MAINTENANCE SOON"
        else:
            risk_level = "LOW"
            urgency = "ROUTINE MONITORING"
        
        # Format model predictions
        model_summary = "\n".join([f"- {model}: {prediction:.1f}" 
                                  for model, prediction in model_predictions.items()])
        
        prompt = f"""
You are an expert aviation maintenance engineer. Analyze this jet engine health data and provide specific maintenance recommendations.

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
- Current risk level analysis
- Immediate safety concerns

2. MAINTENANCE RECOMMENDATIONS
- Specific actions required
- Priority level (Critical/High/Medium/Low)
- Timeline for completion

3. INSPECTION POINTS
- Components to check
- Diagnostic tests to run
- Sensor data to monitor

4. OPERATIONAL GUIDELINES
- Flight restrictions (if any)
- Monitoring requirements
- Follow-up schedule

5. COST & RESOURCE ESTIMATES
- Estimated maintenance hours
- Parts that may be needed
- Personnel requirements

Format as a professional maintenance report with clear sections and actionable items. Prioritize safety above all.
"""
        
        return self.generate_content(prompt)
    
    def get_quick_recommendation(self, health_score: float, predicted_rul: float) -> str:
        """Get quick maintenance recommendation"""
        
        if health_score < 25 or predicted_rul < 20:
            status = "CRITICAL"
            prompt = f"CRITICAL: Engine health {health_score:.1f}%, RUL {predicted_rul:.1f} cycles. Provide immediate safety-critical maintenance actions."
        elif health_score < 50 or predicted_rul < 50:
            status = "WARNING"
            prompt = f"WARNING: Engine health {health_score:.1f}%, RUL {predicted_rul:.1f} cycles. Provide urgent maintenance recommendations."
        else:
            status = "NORMAL"
            prompt = f"NORMAL: Engine health {health_score:.1f}%, RUL {predicted_rul:.1f} cycles. Provide routine maintenance guidelines."
        
        full_prompt = f"""
{prompt}

Keep response concise and actionable. Focus on:
1. Immediate actions needed
2. Maintenance priority
3. Timeline
4. Key components to check
"""
        
        return self.generate_content(full_prompt)
