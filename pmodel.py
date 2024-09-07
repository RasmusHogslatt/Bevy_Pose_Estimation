import numpy as np
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="posenet_mobilenet.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print input details
print("Input Details:")
for input in input_details:
    print(f"  Name: {input['name']}")
    print(f"  Shape: {input['shape']}")
    print(f"  Type: {input['dtype']}")
    print()

# Print output details
print("Output Details:")
for output in output_details:
    print(f"  Name: {output['name']}")
    print(f"  Shape: {output['shape']}")
    print(f"  Type: {output['dtype']}")
    print()

# Optional: Test the model with random input
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Get the output
for output in output_details:
    output_data = interpreter.get_tensor(output['index'])
    print(f"Output {output['name']} shape: {output_data.shape}")