import tensorflow as tf
import os
import h5py
import json

# Load the advanced LSTM model
model_path = "models/lstm_rul_advanced.h5"

print("🔍 Inspecting Advanced LSTM Model...")
print(f"📁 Model path: {model_path}")

try:
    # First, try to inspect the HDF5 file directly
    print(f"\n📊 File Information:")
    file_size = os.path.getsize(model_path)
    print(f"   File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    
    # Inspect HDF5 structure
    print(f"\n🏗️ HDF5 File Structure:")
    print("=" * 80)
    
    with h5py.File(model_path, 'r') as f:
        def print_h5_structure(name, obj):
            print(f"   {name}: {type(obj).__name__}")
            if hasattr(obj, 'attrs'):
                for attr_name, attr_value in obj.attrs.items():
                    print(f"        └─ {attr_name}: {attr_value}")
        
        f.visititems(print_h5_structure)
    
    # Try to load with custom_objects
    print(f"\n🔧 Attempting to Load Model:")
    print("=" * 80)
    
    try:
        # Try loading with custom objects
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
        )
        
        print("✅ Model loaded successfully!")
        
        # Display model architecture
        print(f"\n🏗️ Model Architecture:")
        print("=" * 80)
        model.summary()
        
        # Display model properties
        print(f"\n🔧 Model Properties:")
        print("=" * 80)
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"   Total parameters: {model.count_params():,}")
        
        # Count parameters by type
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
        
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Non-trainable parameters: {non_trainable_params:,}")
        
        # Display layer details
        print(f"\n📋 Layer Details:")
        print("=" * 80)
        for i, layer in enumerate(model.layers):
            layer_type = type(layer).__name__
            output_shape = layer.output_shape
            params = layer.count_params()
            print(f"   Layer {i+1}: {layer_type}")
            print(f"            Output shape: {output_shape}")
            print(f"            Parameters: {params:,}")
            if hasattr(layer, 'units'):
                print(f"            Units: {layer.units}")
            if hasattr(layer, 'activation'):
                print(f"            Activation: {layer.activation.__name__}")
            print()
        
        # Save architecture as JSON for readability
        architecture_json = model.to_json()
        with open("model_architecture.json", "w") as f:
            f.write(architecture_json)
        
        print(f"✅ Model architecture saved to: model_architecture.json")
        
    except Exception as e:
        print(f"❌ Could not load model: {e}")
        
        # Try to extract config from HDF5
        print(f"\n🔍 Extracting Model Config from HDF5:")
        print("=" * 80)
        
        with h5py.File(model_path, 'r') as f:
            if 'model_config' in f.attrs:
                config = f.attrs['model_config']
                config_dict = json.loads(config)
                print("Model Configuration:")
                print(json.dumps(config_dict, indent=2))
            
            if 'training_config' in f.attrs:
                training_config = f.attrs['training_config']
                training_dict = json.loads(training_config)
                print("\nTraining Configuration:")
                print(json.dumps(training_dict, indent=2))
    
    print(f"\n✅ Model inspection completed!")
    
except Exception as e:
    print(f"❌ Error inspecting model: {e}")
    import traceback
    traceback.print_exc()
