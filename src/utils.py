import numpy as np


def calculate_health_index(predicted_rul, max_rul=125):
    """
    Calculate health index percentage from predicted RUL
    
    Args:
        predicted_rul: Predicted remaining useful life
        max_rul: Maximum expected RUL (default: 125)
    
    Returns:
        Health index percentage (0-100)
    """
    # Calculate health index
    health_index = (predicted_rul / max_rul) * 100
    
    # Clamp value between 0 and 100
    health_index = np.clip(health_index, 0, 100)
    
    return float(health_index)


def get_health_status(health_index):
    """
    Get health status based on health index
    
    Args:
        health_index: Health index percentage (0-100)
    
    Returns:
        Tuple of (status_text, color_code)
    """
    if health_index < 25:
        return "Critical", "red"
    elif health_index < 75:
        return "Warning", "orange"
    else:
        return "Healthy", "green"


def format_rul_display(rul_value):
    """
    Format RUL value for display
    
    Args:
        rul_value: RUL value (scalar or array)
    
    Returns:
        Formatted string
    """
    # Convert to scalar if it's an array
    if isinstance(rul_value, np.ndarray):
        rul_value = rul_value.item() if rul_value.size == 1 else rul_value[0]
    
    return f"{float(rul_value):.1f} cycles"


def check_maintenance_alert(rul_value, threshold=20):
    """
    Check if maintenance alert should be shown
    
    Args:
        rul_value: Predicted RUL (scalar or array)
        threshold: Alert threshold in cycles
    
    Returns:
        Boolean indicating if alert should be shown
    """
    # Convert to scalar if it's an array
    if isinstance(rul_value, np.ndarray):
        rul_value = rul_value.item() if rul_value.size == 1 else rul_value[0]
    
    return rul_value < threshold