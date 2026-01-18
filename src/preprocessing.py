import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class DataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.feature_columns = None
        
    def load_data(self, train_path, test_path):
        """Load training and test data from NASA C-MAPSS dataset"""
        # Column names for the dataset
        columns = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
        
        # Load data
        if train_path is not None:
            train_df = pd.read_csv(train_path, sep=r'\s+', header=None, names=columns)
        else:
            train_df = None
            
        if test_path is not None:
            test_df = pd.read_csv(test_path, sep=r'\s+', header=None, names=columns)
        else:
            test_df = None
            
        return train_df, test_df
    
    def calculate_rul(self, df):
        """Calculate RUL for training data: RUL = Max(Cycle) - Current(Cycle) per unit"""
        rul_df = df.copy()
        rul_df['RUL'] = rul_df.groupby('unit')['cycle'].transform(lambda x: x.max() - x)
        return rul_df
    
    def drop_constant_sensors(self, df):
        """Drop constant sensors that hold no value in FD001"""
        constant_sensors = ['s1', 's5', 's10', 's16', 's18', 's19']
        
        # Keep only sensor columns and operational settings
        sensor_cols = [col for col in df.columns if col.startswith('s')]
        op_cols = ['op1', 'op2', 'op3']
        
        # Filter out constant sensors
        remaining_sensors = [s for s in sensor_cols if s not in constant_sensors]
        self.feature_columns = op_cols + remaining_sensors
        
        return df[self.feature_columns + ['unit', 'cycle', 'RUL']]
    
    def apply_scaling(self, df):
        """Apply MinMaxScaler to the remaining sensors"""
        # Fit scaler on training data features
        feature_data = df[self.feature_columns]
        self.scaler.fit(feature_data)
        
        # Transform features
        scaled_features = self.scaler.transform(feature_data)
        df_scaled = df.copy()
        df_scaled[self.feature_columns] = scaled_features
        
        return df_scaled
    
    def gen_sequence(self, data, seq_length, cols):
        """Generate sequences for LSTM: reshape 2D DataFrames into 3D arrays"""
        sequences = []
        labels = []
        
        for unit in data['unit'].unique():
            unit_data = data[data['unit'] == unit].sort_values('cycle')
            
            # Create sequences
            for i in range(len(unit_data) - seq_length + 1):
                seq = unit_data[cols].iloc[i:i + seq_length].values
                label = unit_data['RUL'].iloc[i + seq_length - 1]
                
                sequences.append(seq)
                labels.append(label)
        
        return np.array(sequences), np.array(labels)
    
    def process_training_data(self, train_path, seq_length=50):
        """Complete pipeline for processing training data"""
        # Load data
        train_df = self.load_data(train_path, None)[0]
        
        # Calculate RUL
        train_df = self.calculate_rul(train_df)
        
        # Drop constant sensors
        train_df = self.drop_constant_sensors(train_df)
        
        # Apply scaling
        train_df = self.apply_scaling(train_df)
        
        # Generate sequences
        X, y = self.gen_sequence(train_df, seq_length, self.feature_columns)
        
        return X, y, train_df
    
    def process_test_data(self, test_path, seq_length=50):
        """Process test data for prediction"""
        # Load data
        _, test_df = self.load_data(None, test_path)
        
        # For test data, we need to create dummy RUL for sequence generation
        test_df['RUL'] = 0  # Placeholder
        
        # Drop constant sensors
        test_df = self.drop_constant_sensors(test_df)
        
        # Apply scaling using fitted scaler
        test_df[self.feature_columns] = self.scaler.transform(test_df[self.feature_columns])
        
        return test_df