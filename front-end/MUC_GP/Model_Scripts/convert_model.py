
import tensorflow as tf
import os
import sys

def convert_keras_to_tflite():
    try:
        # Print working directory and TensorFlow version
        print(f"Current directory: {os.getcwd()}")
        print(f"TensorFlow version: {tf.__version__}")
        
        # Load keras model
        print("\nStep 1: Loading Keras model...")
        model_name = 'har_model.keras'  # Update this to your model's name
        if not os.path.exists(model_name):
            print(f"❌ Error: Could not find {model_name} in current directory")
            sys.exit(1)
            
        model = tf.keras.models.load_model(model_name)
        print("✅ Model loaded successfully")
        
        # Print model summary
        print("\nModel Summary:")
        model.summary()
        
        # Convert to TFLite
        print("\nStep 2: Converting to TFLite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        print("✅ Conversion successful")
        
        # Save the TFLite model
        print("\nStep 3: Saving TFLite model...")
        tflite_name = 'model.tflite'
        with open(tflite_name, 'wb') as f:
            f.write(tflite_model)
        print(f"✅ Model saved as '{tflite_name}'")
        
        # Verify the converted model
        print("\nStep 4: Verifying converted model...")
        interpreter = tf.lite.Interpreter(model_path=tflite_name)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("\nModel Details:")
        print(f"Input shape: {input_details[0]['shape']}")
        print(f"Input type: {input_details[0]['dtype']}")
        print(f"Output shape: {output_details[0]['shape']}")
        print(f"Output type: {output_details[0]['dtype']}")
        
        print("\n✅ Conversion complete! You can now add 'model.tflite' to your Xcode project")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    convert_keras_to_tflite()
