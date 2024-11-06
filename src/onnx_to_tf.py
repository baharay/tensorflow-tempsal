import onnx
from onnx_tf.backend import prepare

# Load the ONNX model
onnx_model = onnx.load("modified_model.onnx")

# Convert the ONNX model to TensorFlow
tf_rep = prepare(onnx_model)

# Export the TensorFlow model as a TensorFlow SavedModel
tf_rep.export_graph("model_tf")
