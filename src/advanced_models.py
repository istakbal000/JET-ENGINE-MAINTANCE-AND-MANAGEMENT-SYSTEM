import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GRU, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input, SimpleRNN
from scipy.spatial.distance import euclidean
from scipy.stats import expon


class ExponentialDegradationModel:
    """Exponential degradation model for RUL prediction"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def fit(self, X_train, y_train):
        """Fit exponential degradation model"""
        # For simplicity, use linear regression on log-transformed RUL
        # In practice, this would be more sophisticated exponential fitting
        
        # Reshape 3D to 2D for StandardScaler
        X_2d = X_train.reshape(X_train.shape[0], -1)
        self.scaler.fit(X_2d)
        X_scaled = self.scaler.transform(X_2d)
        
        # Use last cycle values for exponential fitting
        last_cycle_features = X_scaled[:, -len(self.scaler.scale_):]  # Get last time step
        
        self.model = LinearRegression()
        self.model.fit(last_cycle_features, y_train)
        
    def predict(self, X_test):
        """Predict RUL using exponential degradation"""
        X_2d = X_test.reshape(X_test.shape[0], -1)
        X_scaled = self.scaler.transform(X_2d)
        n_features = len(self.scaler.scale_)
        last_cycle_features = X_scaled[:, -n_features:]  # Get last time step
        
        rul_pred = self.model.predict(last_cycle_features)
        # Apply exponential decay factor
        rul_pred = rul_pred * np.exp(-0.01 * np.arange(len(rul_pred)))
        
        return np.maximum(rul_pred, 0)  # Ensure non-negative


class SimilarityBasedModel:
    """Similarity-based model implemented as an SGDRegressor over flattened sequences"""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, X_train, y_train, epochs=50):
        """Train an SGDRegressor on flattened, scaled sequences

        Parameters
        - X_train: ndarray, shape (n_samples, seq_length, n_features)
        - y_train: ndarray, shape (n_samples,) or (n_samples, 1)
        - epochs: int, passed to SGDRegressor.max_iter
        """
        X_flat = X_train.reshape(X_train.shape[0], -1)
        self.scaler.fit(X_flat)
        X_scaled = self.scaler.transform(X_flat)

        self.model = SGDRegressor(max_iter=epochs, tol=1e-3, random_state=42)
        # sklearn expects 1D targets
        y_train_flat = y_train.ravel() if y_train.ndim > 1 else y_train
        self.model.fit(X_scaled, y_train_flat)

    def predict(self, X_test):
        """Predict RUL using the trained regressor"""
        X_flat = X_test.reshape(X_test.shape[0], -1)
        X_scaled = self.scaler.transform(X_flat)
        preds = self.model.predict(X_scaled)
        return np.maximum(preds, 0)


class LSTMRULModel:
    """LSTM model for RUL prediction (enhanced version)"""
    
    def __init__(self, seq_length=50, n_features=18):
        self.seq_length = seq_length
        self.n_features = n_features
        self.model = None
        
    def create_model(self):
        """Create enhanced LSTM model"""
        model = Sequential([
            GRU(128, return_sequences=True, activation='tanh', input_shape=(self.seq_length, self.n_features)),
            Dropout(0.3),
            GRU(64, return_sequences=True, activation='tanh'),
            Dropout(0.3),
            GRU(32, return_sequences=False, activation='tanh'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='relu')
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def fit(self, X_train, y_train, epochs=30, batch_size=128, validation_split=0.2):
        """Train LSTM model"""
        self.model = self.create_model()
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                     validation_split=validation_split, verbose=1)
    
    def predict(self, X_test):
        """Predict RUL"""
        return self.model.predict(X_test, verbose=0).flatten()


class LSTMClassificationModel:
    """LSTM model for binary and multiclass classification"""
    
    def __init__(self, seq_length=50, n_features=18, n_classes=3):
        self.seq_length = seq_length
        self.n_features = n_features
        self.n_classes = n_classes
        self.model = None
        
    def create_model(self):
        """Create LSTM classification model"""
        output_units = 1 if self.n_classes == 2 else self.n_classes
        output_activation = 'sigmoid' if self.n_classes == 2 else 'softmax'
        loss_fn = 'binary_crossentropy' if self.n_classes == 2 else 'categorical_crossentropy'

        model = Sequential([
            GRU(64, return_sequences=True, activation='tanh', input_shape=(self.seq_length, self.n_features)),
            Dropout(0.3),
            GRU(32, return_sequences=False, activation='tanh'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(output_units, activation=output_activation)
        ])
        
        model.compile(
            optimizer='adam',
            loss=loss_fn,
            metrics=['accuracy'],
        )
        return model
    
    def fit(self, X_train, y_train, epochs=30, batch_size=128, validation_split=0.2):
        """Train LSTM classification model"""
        self.model = self.create_model()
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                     validation_split=validation_split, verbose=1)
    
    def predict(self, X_test):
        """Predict class probabilities"""
        return self.model.predict(X_test, verbose=0)


class RNNClassificationModel:
    """RNN model for binary and multiclass classification"""
    
    def __init__(self, seq_length=50, n_features=18, n_classes=3):
        self.seq_length = seq_length
        self.n_features = n_features
        self.n_classes = n_classes
        self.model = None
        
    def create_model(self):
        """Create RNN classification model"""
        output_units = 1 if self.n_classes == 2 else self.n_classes
        output_activation = 'sigmoid' if self.n_classes == 2 else 'softmax'
        loss_fn = 'binary_crossentropy' if self.n_classes == 2 else 'categorical_crossentropy'

        model = Sequential([
            SimpleRNN(64, return_sequences=True, activation='tanh', input_shape=(self.seq_length, self.n_features)),
            Dropout(0.3),
            SimpleRNN(32, return_sequences=False, activation='tanh'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(output_units, activation=output_activation)
        ])
        
        model.compile(
            optimizer='adam',
            loss=loss_fn,
            metrics=['accuracy'],
        )
        return model
    
    def fit(self, X_train, y_train, epochs=30, batch_size=128, validation_split=0.2):
        """Train RNN classification model"""
        self.model = self.create_model()
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                     validation_split=validation_split, verbose=1)
    
    def predict(self, X_test):
        """Predict class probabilities"""
        return self.model.predict(X_test, verbose=0)


class CNN1DClassificationModel:
    """1D CNN model for binary and multiclass classification"""
    
    def __init__(self, seq_length=50, n_features=18, n_classes=3):
        self.seq_length = seq_length
        self.n_features = n_features
        self.n_classes = n_classes
        self.model = None
        
    def create_model(self):
        """Create 1D CNN classification model"""
        output_units = 1 if self.n_classes == 2 else self.n_classes
        output_activation = 'sigmoid' if self.n_classes == 2 else 'softmax'
        loss_fn = 'binary_crossentropy' if self.n_classes == 2 else 'categorical_crossentropy'

        model = Sequential([
            Conv1D(64, kernel_size=3, activation='relu', input_shape=(self.seq_length, self.n_features)),
            Conv1D(64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            Conv1D(32, kernel_size=3, activation='relu'),
            Conv1D(32, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(output_units, activation=output_activation)
        ])
        
        model.compile(
            optimizer='adam',
            loss=loss_fn,
            metrics=['accuracy'],
        )
        return model
    
    def fit(self, X_train, y_train, epochs=30, batch_size=128, validation_split=0.2):
        """Train 1D CNN classification model"""
        self.model = self.create_model()
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                     validation_split=validation_split, verbose=1)
    
    def predict(self, X_test):
        """Predict class probabilities"""
        return self.model.predict(X_test, verbose=0)


class CNN1DSVMModel:
    """1D CNN-SVM hybrid model for binary classification"""
    
    def __init__(self, seq_length=50, n_features=18):
        self.seq_length = seq_length
        self.n_features = n_features
        self.cnn_model = None
        self.svm_model = None
        self.scaler = StandardScaler()
        
    def create_cnn_feature_extractor(self):
        """Create CNN for feature extraction"""
        model = Sequential([
            Conv1D(32, kernel_size=3, activation='relu', input_shape=(self.seq_length, self.n_features)),
            Conv1D(32, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            Conv1D(16, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            Flatten(),
            Dense(64, activation='relu')
        ])
        return model
    
    def fit(self, X_train, y_train, epochs=20, batch_size=128):
        """Train CNN-SVM hybrid model"""
        # Train CNN for feature extraction
        self.cnn_model = self.create_cnn_feature_extractor()
        
        # Extract features
        features = self.cnn_model.predict(X_train, verbose=0)
        
        # Scale features
        self.scaler.fit(features)
        features_scaled = self.scaler.transform(features)
        
        # Train SVM
        self.svm_model = SVC(kernel='rbf', probability=True, random_state=42)
        self.svm_model.fit(features_scaled, y_train)
    
    def predict(self, X_test):
        """Predict using CNN-SVM hybrid"""
        # Extract features
        features = self.cnn_model.predict(X_test, verbose=0)
        features_scaled = self.scaler.transform(features)
        
        # Predict with SVM
        return self.svm_model.predict_proba(features_scaled)


def convert_rul_to_classes(rul_values, n_classes=3):
    """Convert RUL values to classification labels"""
    if n_classes == 2:
        # Binary: Healthy (RUL > 30) vs Warning (RUL <= 30)
        return (rul_values <= 30).astype(int)
    else:
        # Multiclass: Healthy (RUL > 60), Warning (30 < RUL <= 60), Critical (RUL <= 30)
        classes = np.zeros(len(rul_values), dtype=int)
        classes[rul_values <= 30] = 2  # Critical
        classes[(rul_values > 30) & (rul_values <= 60)] = 1  # Warning
        classes[rul_values > 60] = 0  # Healthy
        return classes


def evaluate_models(models, X_test, y_test):
    """Evaluate multiple models and return performance metrics"""
    results = {}
    
    for name, model in models.items():
        if 'classification' in name.lower():
            # For classification models
            y_pred = model.predict(X_test)
            y_pred_classes = np.argmax(y_pred, axis=1) if len(y_pred.shape) > 1 else (y_pred > 0.5).astype(int)
            
            accuracy = accuracy_score(y_test, y_pred_classes)
            results[name] = {'accuracy': accuracy}
        else:
            # For regression models
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            results[name] = {'mse': mse, 'mae': mae}
    
    return results
