import tensorflow as tf

# Load the TensorFlow model
model = tf.saved_model.load("model_tf")
print(model.signatures)

# Prepare an input (adjust to match your model's input shape)
input_tensor = tf.random.normal([1, 3, 256, 256])

# Perform inference
output = model(input=input_tensor)
print(output)
