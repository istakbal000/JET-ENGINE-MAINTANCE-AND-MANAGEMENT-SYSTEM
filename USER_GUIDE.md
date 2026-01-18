# 📖 User Guide - Jet Engine Predictive Maintenance System

## 🎯 Welcome to Your AI-Powered Maintenance Assistant!

This guide will help you understand and use the jet engine predictive maintenance system effectively. Whether you're an aviation engineer, maintenance technician, or fleet manager, this system provides intelligent insights to keep your engines running safely and efficiently.

## 🚀 Getting Started

### Step 1: Launch the Application
```bash
# Open terminal/command prompt
cd jet_engine_maintenance
streamlit run app_advanced.py
```

### Step 2: Open Your Browser
Navigate to: **http://localhost:8501**

You'll see the main dashboard with:
- **Health monitoring gauges**
- **Model selection options**
- **AI maintenance advisor**
- **Interactive charts**

## 📊 Understanding the Dashboard

### Main Dashboard Layout

```
┌─────────────────────────────────────────────────────────────┐
│  🚀 Jet Engine Predictive Maintenance System              │
├─────────────────────────────────────────────────────────────┤
│  ⚙️ Configuration Sidebar                                  │
│  ├─ Select Unit (dropdown)                                 │
│  ├─ Select Model (7 options)                              │
│  ├─ AI Advisor Settings                                    │
│  └─ Enable/Disable AI                                      │
├─────────────────────────────────────────────────────────────┤
│  📊 Engine Health Analysis                                 │
│  ├─ Health Score Gauge                                     │
│  ├─ Predicted RUL                                         │
│  ├─ Status Indicator                                       │
│  └─ Maintenance Alerts                                     │
├─────────────────────────────────────────────────────────────┤
│  📈 Model Performance Comparison                            │
│  └─ Bar chart showing all 7 model results                 │
├─────────────────────────────────────────────────────────────┤
│  📈 Sensor Trends                                          │
│  └─ Time-series plots of key sensors                      │
├─────────────────────────────────────────────────────────────┤
│  🤖 AI Maintenance Advisor                                 │
│  ├─ Get AI Recommendations button                          │
│  ├─ Quick Recommendation button                            │
│  └─ Professional maintenance reports                       │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 How to Use the System

### 1. Select Your Engine Unit

**What it is**: Each "unit" represents a specific jet engine in your fleet.

**How to use**:
1. Look at the sidebar (left side)
2. Click the "Select Unit" dropdown
3. Choose the engine you want to analyze
4. The system shows available cycles for that engine

**What you'll see**:
- Unit number (e.g., "Unit 1", "Unit 2")
- Available cycles (e.g., "Unit 1: 192 cycles available")

### 2. Choose Your Prediction Model

**What it is**: We have 7 different AI models, each with unique strengths.

**Model Options**:
| Model | Best For | Description |
|--------|-----------|-------------|
| **Exponential Degradation** | Quick estimates | Mathematical decay model |
| **Similarity-Based** | Pattern matching | Compares to historical data |
| **LSTM RUL** | Temporal analysis | Best for time-series data |
| **LSTM Binary** | Yes/No decisions | Healthy vs Warning |
| **RNN Multiclass** | Detailed health | Healthy/Warning/Critical |
| **CNN Multiclass** | Feature detection | Advanced pattern recognition |
| **CNN-SVM Binary** | High accuracy | Hybrid approach |

**How to choose**:
- **For quick decisions**: Use Exponential Degradation
- **For detailed analysis**: Use LSTM RUL
- **For health classification**: Use RNN Multiclass
- **For highest accuracy**: Use CNN-SVM Binary

### 3. Analyze Engine Health

**Health Score Gauge**:
- 🟢 **75-100%**: Good health, routine monitoring
- 🟡 **50-74%**: Moderate health, schedule maintenance
- 🟠 **25-49%**: Poor health, urgent maintenance needed
- 🔴 **0-24%**: Critical health, immediate action required

**Predicted RUL (Remaining Useful Life)**:
- Shows cycles remaining before maintenance
- **< 20 cycles**: 🔴 Immediate maintenance required
- **< 50 cycles**: 🟡 Schedule maintenance soon
- **≥ 50 cycles**: 🟢 Normal operation

### 4. Review Model Performance

**Model Comparison Chart**:
- Shows results from all 7 models
- Higher bars = better performance
- Helps identify consensus among models
- Click legend to show/hide models

**Understanding Results**:
- **Consistent results**: High confidence in prediction
- **Varied results**: Consider multiple models or AI advice
- **Outliers**: May indicate unusual engine behavior

### 5. Monitor Sensor Trends

**What you're seeing**: Real sensor data over time
- **Temperature sensors**: Engine heat patterns
- **Pressure sensors**: System pressure changes
- **Vibration sensors**: Mechanical wear indicators
- **Flow sensors**: Fluid system performance

**How to interpret**:
- **Stable patterns**: Normal operation
- **Gradual changes**: Expected wear
- **Sudden spikes**: Potential issues
- **Erratic behavior**: Requires investigation

## 🤖 Using the AI Maintenance Advisor

### Enable AI Assistant

1. **Check the sidebar** for "AI Advisor Settings"
2. **Verify API key** is loaded (shows "✅ AI Advisor Connected")
3. **Enable "AI Maintenance Advisor"** checkbox

### Get Detailed AI Recommendations

**When to use**: For comprehensive maintenance planning

**What you get**:
1. **Risk Assessment** - Current danger level analysis
2. **Maintenance Recommendations** - Specific actions needed
3. **Inspection Points** - Components to check
4. **Operational Guidelines** - Flight restrictions if any
5. **Cost Estimates** - Resources and time required

**How to use**:
1. Click "🧠 Get AI Maintenance Recommendations"
2. Wait for AI analysis (takes 10-30 seconds)
3. Read the professional maintenance report
4. Follow the recommended actions

### Get Quick Recommendations

**When to use**: For urgent, fast decisions

**What you get**:
- Immediate actions needed
- Maintenance priority level
- Timeline for completion
- Key components to check

**How to use**:
1. Click "🚨 Get Quick Recommendation"
2. Receive fast, actionable advice
3. Follow the urgent recommendations

## 📊 Understanding AI Recommendations

### Risk Levels

**🔴 CRITICAL**
- Health score < 25% OR RUL < 20 cycles
- **Immediate action required**
- May require grounding the aircraft
- Emergency maintenance procedures

**🟠 HIGH**
- Health score 25-49% OR RUL 20-49 cycles
- **Urgent maintenance needed**
- Schedule within 24-48 hours
- Limited operation recommended

**🟡 MODERATE**
- Health score 50-74% OR RUL 50-99 cycles
- **Schedule maintenance soon**
- Plan within 1-2 weeks
- Normal operation allowed

**🟢 LOW**
- Health score 75-100% AND RUL ≥ 100 cycles
- **Routine monitoring**
- Next scheduled maintenance
- Full operation permitted

### Maintenance Actions

**Immediate Actions** (Critical/High Risk):
- Inspect specific components
- Run diagnostic tests
- Monitor sensor data
- Consider grounding aircraft

**Scheduled Actions** (Moderate Risk):
- Plan maintenance window
- Order replacement parts
- Schedule qualified technicians
- Prepare maintenance documentation

**Routine Actions** (Low Risk):
- Continue normal monitoring
- Log current readings
- Plan next inspection
- Update maintenance records

## 🎯 Real-World Scenarios

### Scenario 1: Routine Check
**Situation**: Unit 15 shows 85% health, 150 cycles RUL
**Actions**:
1. Select Unit 15
2. Choose any model (LSTM RUL recommended)
3. Review stable sensor trends
4. Get AI recommendations for routine planning
5. Schedule next maintenance

### Scenario 2: Warning Signs
**Situation**: Unit 23 shows 45% health, 35 cycles RUL
**Actions**:
1. Select Unit 23
2. Use CNN-SVM Binary for classification
3. Check sensor trends for anomalies
4. Get AI recommendations for urgent planning
5. Schedule maintenance within 48 hours

### Scenario 3: Critical Alert
**Situation**: Unit 7 shows 15% health, 8 cycles RUL
**Actions**:
1. Select Unit 7
2. Use multiple models for consensus
3. Analyze sensor trends for failure patterns
4. Get immediate AI recommendations
5. Consider grounding aircraft
6. Emergency maintenance procedures

## 📈 Best Practices

### Daily Operations
- **Check high-risk units first** (low health scores)
- **Review sensor trends** for anomalies
- **Compare multiple models** for consensus
- **Document findings** in maintenance logs

### Weekly Reviews
- **Analyze all units** systematically
- **Update maintenance schedules** based on predictions
- **Review AI recommendations** for planning
- **Track prediction accuracy** over time

### Monthly Planning
- **Evaluate model performance** trends
- **Update maintenance calendars**
- **Review resource requirements**
- **Plan for seasonal variations**

## 🚨 Troubleshooting Common Issues

### "No units with sufficient data found"
**Cause**: All units need at least 50 cycles of data
**Solution**: 
- Check if data files are loaded correctly
- Verify data files exist in `data/` directory
- Use training data if test data is insufficient

### "AI Advisor Error"
**Cause**: API key issue or network problem
**Solution**:
- Check `.env` file contains valid API key
- Verify internet connection
- Try "Quick Recommendation" first

### "Model prediction failed"
**Cause**: Model files missing or corrupted
**Solution**:
- Run `python train_advanced.py` to train models
- Check `models/` directory for `.h5` files
- Restart the application

### "Charts not displaying"
**Cause**: Browser or JavaScript issue
**Solution**:
- Refresh the browser page
- Clear browser cache
- Try a different browser

## 📞 Getting Help

### Self-Service Resources
- **README.md**: Technical documentation
- **ARCHITECTURE.md**: System design details
- **Code comments**: Inline explanations
- **Error messages**: Descriptive troubleshooting

### When to Contact Support
- **System crashes**: Repeated application failures
- **Data issues**: Missing or corrupted data
- **API problems**: Persistent AI advisor failures
- **Performance issues**: Slow response times

### Information to Provide
- **Error message**: Full error text
- **Steps taken**: What you were doing
- **Unit number**: Which engine you were analyzing
- **Model used**: Which prediction model was selected

## 🎓 Tips for Advanced Users

### Model Selection Strategy
- **Use ensemble approach**: Compare multiple models
- **Consider model strengths**: Match model to use case
- **Track accuracy**: Note which models work best for your fleet
- **Seasonal adjustments**: Different models for different conditions

### AI Prompt Optimization
- **Provide context**: Include operational conditions
- **Specify urgency**: Mention time constraints
- **Request specifics**: Ask for particular components
- **Follow up**: Ask for clarification if needed

### Data Analysis Techniques
- **Trend analysis**: Look for gradual changes
- **Anomaly detection**: Identify unusual patterns
- **Comparative analysis**: Compare similar units
- **Historical patterns**: Learn from past failures

---

## 🎉 Conclusion

This system combines cutting-edge AI with practical aviation maintenance expertise to help you:

✅ **Predict failures before they happen**
✅ **Optimize maintenance schedules**
✅ **Reduce downtime and costs**
✅ **Improve safety and reliability**
✅ **Make data-driven decisions**

**Remember**: The AI advisor is a tool to support your professional judgment, not replace it. Always combine AI recommendations with your expertise and experience.

Happy maintaining! 🚀✈️
