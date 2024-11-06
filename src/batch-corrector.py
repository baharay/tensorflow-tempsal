import onnx
from onnx import version_converter, helper

# Load the ONNX model
model = onnx.load("model.onnx")

# Find and modify the resize operation
for node in model.graph.node:
    if node.op_type == "Resize":
        for attr in node.attribute:
            if attr.name == "mode" and attr.s.decode('utf-8') == 'cubic':
                attr.s = b'linear'  # Change cubic to linear or nearest

# Save the modified ONNX model
onnx.save(model, "modified_model.onnx")
