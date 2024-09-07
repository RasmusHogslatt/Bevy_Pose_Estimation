import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="1.tflite")
interpreter.allocate_tensors()

# Print all operation details
for op in interpreter._get_ops_details():
    print(op)