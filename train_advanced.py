import os
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from src.preprocessing import DataProcessor
from src.advanced_models import (
    ExponentialDegradationModel, SimilarityBasedModel, LSTMRULModel,
    LSTMClassificationModel, RNNClassificationModel, CNN1DClassificationModel,
    CNN1DSVMModel, convert_rul_to_classes, evaluate_models
)


def main():
    """Train all 7 advanced models"""
    # Sequence length (cycles) used for LSTM/CNN sequence input
    SEQ_LENGTH = 50
    print("🚀 Starting Advanced Jet Engine RUL Model Training...")
    
    # File paths
    train_path = "data/train_FD001.txt"
    test_path = "data/test_FD001.txt"
    model_save_path = "models/"
    
    # Ensure models directory exists
    os.makedirs(model_save_path, exist_ok=True)
    
    # Initialize data processor
    processor = DataProcessor()
    
    # Process training data
    print("📊 Processing training data...")
    X_train, y_train, train_df = processor.process_training_data(train_path, seq_length=SEQ_LENGTH)
    
    print(f"✅ Training data shape: {X_train.shape}")
    print(f"✅ Training labels shape: {y_train.shape}")
    print(f"✅ Feature columns: {len(processor.feature_columns)}")
    
    # Split data for evaluation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Convert RUL to classes for classification models
    y_train_binary = convert_rul_to_classes(y_train_split, n_classes=2).reshape(-1, 1)
    y_train_multi = convert_rul_to_classes(y_train_split, n_classes=3)
    y_val_binary = convert_rul_to_classes(y_val_split, n_classes=2).reshape(-1, 1)
    y_val_multi = convert_rul_to_classes(y_val_split, n_classes=3)
    
    # Convert to categorical for multiclass
    y_train_multi_cat = to_categorical(y_train_multi, num_classes=3)
    y_val_multi_cat = to_categorical(y_val_multi, num_classes=3)
    
    print("\n" + "="*60)
    print("🔬 TRAINING MODEL 1: EXPONENTIAL DEGRADATION")
    print("="*60)
    
    # Model 1: Exponential Degradation
    exp_model = ExponentialDegradationModel()
    exp_model.fit(X_train_split, y_train_split)
    exp_pred = exp_model.predict(X_val_split)
    exp_mse = np.mean((exp_pred - y_val_split)**2)
    exp_mae = np.mean(np.abs(exp_pred - y_val_split))
    print(f"✅ Exponential Model - MSE: {exp_mse:.4f}, MAE: {exp_mae:.4f}")
    
    print("\n" + "="*60)
    print("🔍 TRAINING MODEL 2: SIMILARITY-BASED")
    print("="*60)
    
    # Model 2: Similarity-based
    sim_model = SimilarityBasedModel()
    sim_model.fit(X_train_split, y_train_split, epochs=50)
    sim_pred = sim_model.predict(X_val_split)
    sim_mse = np.mean((sim_pred - y_val_split)**2)
    sim_mae = np.mean(np.abs(sim_pred - y_val_split))
    print(f"✅ Similarity Model - MSE: {sim_mse:.4f}, MAE: {sim_mae:.4f}")
    
    print("\n" + "="*60)
    print("🧠 TRAINING MODEL 3: LSTM FOR RUL PREDICTION")
    print("="*60)
    
    # Model 3: LSTM for RUL
    lstm_rul_model = LSTMRULModel(seq_length=SEQ_LENGTH, n_features=len(processor.feature_columns))
    lstm_rul_model.fit(X_train_split, y_train_split, epochs=50, batch_size=128)
    lstm_rul_pred = lstm_rul_model.predict(X_val_split)
    lstm_rul_mse = np.mean((lstm_rul_pred - y_val_split)**2)
    lstm_rul_mae = np.mean(np.abs(lstm_rul_pred - y_val_split))
    print(f"✅ LSTM RUL Model - MSE: {lstm_rul_mse:.4f}, MAE: {lstm_rul_mae:.4f}")
    
    print("\n" + "="*60)
    print("🏷️ TRAINING MODEL 4: LSTM FOR BINARY CLASSIFICATION")
    print("="*60)
    
    # Model 4: LSTM for Binary Classification
    lstm_binary_model = LSTMClassificationModel(seq_length=SEQ_LENGTH, n_features=len(processor.feature_columns), n_classes=2)
    lstm_binary_model.fit(X_train_split, y_train_binary, epochs=50, batch_size=128)
    lstm_binary_pred = lstm_binary_model.predict(X_val_split)
    lstm_binary_pred_labels = (lstm_binary_pred.ravel() > 0.5).astype(int)
    lstm_binary_acc = np.mean(lstm_binary_pred_labels.reshape(-1, 1) == y_val_binary)
    print(f"✅ LSTM Binary Model - Accuracy: {lstm_binary_acc:.4f}")
    
    print("\n" + "="*60)
    print("🏷️ TRAINING MODEL 5: RNN FOR MULTICLASS CLASSIFICATION")
    print("="*60)
    
    # Model 5: RNN for Multiclass Classification
    rnn_multi_model = RNNClassificationModel(seq_length=SEQ_LENGTH, n_features=len(processor.feature_columns), n_classes=3)
    rnn_multi_model.fit(X_train_split, y_train_multi_cat, epochs=50, batch_size=128)
    rnn_multi_pred = rnn_multi_model.predict(X_val_split)
    rnn_multi_acc = np.mean(np.argmax(rnn_multi_pred, axis=1) == y_val_multi)
    print(f"✅ RNN Multiclass Model - Accuracy: {rnn_multi_acc:.4f}")
    
    print("\n" + "="*60)
    print("🏷️ TRAINING MODEL 6: 1D CNN FOR MULTICLASS CLASSIFICATION")
    print("="*60)
    
    # Model 6: 1D CNN for Multiclass Classification
    cnn_multi_model = CNN1DClassificationModel(seq_length=SEQ_LENGTH, n_features=len(processor.feature_columns), n_classes=3)
    cnn_multi_model.fit(X_train_split, y_train_multi_cat, epochs=50, batch_size=128)
    cnn_multi_pred = cnn_multi_model.predict(X_val_split)
    cnn_multi_acc = np.mean(np.argmax(cnn_multi_pred, axis=1) == y_val_multi)
    print(f"✅ CNN Multiclass Model - Accuracy: {cnn_multi_acc:.4f}")
    
    print("\n" + "="*60)
    print("🏷️ TRAINING MODEL 7: 1D CNN-SVM FOR BINARY CLASSIFICATION")
    print("="*60)
    
    # Model 7: 1D CNN-SVM for Binary Classification
    cnn_svm_model = CNN1DSVMModel(seq_length=SEQ_LENGTH, n_features=len(processor.feature_columns))
    cnn_svm_model.fit(X_train_split, y_train_binary, epochs=50)
    cnn_svm_pred = cnn_svm_model.predict(X_val_split)
    cnn_svm_acc = np.mean(np.argmax(cnn_svm_pred, axis=1).reshape(-1, 1) == y_val_binary)
    print(f"✅ CNN-SVM Binary Model - Accuracy: {cnn_svm_acc:.4f}")
    
    # Save models
    print("\n" + "="*60)
    print("💾 SAVING MODELS")
    print("="*60)
    
    # Save LSTM RUL model (main model)
    lstm_rul_model.model.save(os.path.join(model_save_path, "lstm_rul_advanced.h5"))
    print("✅ LSTM RUL model saved")
    
    # Create results summary
    print("\n" + "="*60)
    print("📊 MODEL COMPARISON SUMMARY")
    print("="*60)
    
    results = {
        "Exponential Degradation": {"MAE": exp_mae, "MSE": exp_mse},
        "Similarity-Based": {"MAE": sim_mae, "MSE": sim_mse},
        "LSTM RUL": {"MAE": lstm_rul_mae, "MSE": lstm_rul_mse},
        "LSTM Binary": {"Accuracy": lstm_binary_acc},
        "RNN Multiclass": {"Accuracy": rnn_multi_acc},
        "CNN Multiclass": {"Accuracy": cnn_multi_acc},
        "CNN-SVM Binary": {"Accuracy": cnn_svm_acc}
    }
    
    # Print comparison table
    print("\nModel Performance Comparison:")
    print("-" * 50)
    for model_name, metrics in results.items():
        if "Accuracy" in metrics:
            print(f"{model_name:25} | Accuracy: {metrics['Accuracy']:.4f} ({metrics['Accuracy']*100:.2f}%)")
        else:
            print(f"{model_name:25} | MAE: {metrics['MAE']:8.4f} | MSE: {metrics['MSE']:8.4f}")
    
    # Find best models
    best_rul_model = min([k for k in results.keys() if "MAE" in results[k]], 
                        key=lambda x: results[x]["MAE"])
    best_classification_model = max([k for k in results.keys() if "Accuracy" in results[k]], 
                                key=lambda x: results[x]["Accuracy"])
    
    print(f"\n🏆 Best RUL Model: {best_rul_model} (MAE: {results[best_rul_model]['MAE']:.4f})")
    print(f"🏆 Best Classification Model: {best_classification_model} (Accuracy: {results[best_classification_model]['Accuracy']:.4f})")
    
    print("\n🎉 Advanced training completed successfully!")
    print(f"📁 Models saved to: {model_save_path}")


if __name__ == "__main__":
    main()
