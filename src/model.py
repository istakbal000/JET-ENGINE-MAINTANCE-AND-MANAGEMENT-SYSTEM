import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense


def create_model(input_shape):
    """
    Create GRU model for RUL prediction (replaces previous LSTM)
    
    Args:
        input_shape: tuple of (sequence_length, n_features)
    
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # Layer 1: GRU (128 units, return_sequences=True, activation='tanh')
        GRU(128, return_sequences=True, activation='tanh', input_shape=input_shape),
        
        # Layer 2: Dropout (0.2)
        Dropout(0.2),
        
        # Layer 3: GRU (64 units, return_sequences=False)
        GRU(64, return_sequences=False),
        
        # Layer 4: Dropout (0.2)
        Dropout(0.2),
        
        # Output: Dense (1 unit, activation='relu')
        Dense(1, activation='relu')  # ReLU because RUL cannot be negative
    ])
    
    # Compile with adam optimizer and mean_squared_error loss
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
    return model


def get_model_summary(model):
    """Get model summary as string"""
    import io
    import sys
    
    # Capture model summary
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    model.summary()
    sys.stdout = old_stdout
    
    return buffer.getvalue()