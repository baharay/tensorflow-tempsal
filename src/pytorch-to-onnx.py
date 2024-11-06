import torch
import torch.onnx
from collections import OrderedDict
from model import PNASBoostedModelMultiLevel
from model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths to the model checkpoints
model_checkpoint_path = "./checkpoints/multilevel_tempsal.pt"
time_slices = 5
train_model = 0

from model import PNASBoostedModelMultiLevel
model = PNASBoostedModelMultiLevel(device, model_checkpoint_path, model_checkpoint_path, time_slices, train_model=train_model )    
    
# Load model    
#model = load_model(model_checkpoint_path)
model = model.to(device)
model.eval()



dummy_input = torch.randn(1, 3, 256, 256)#.cuda()  # Modify this as needed


# # Export the model to ONNX
# torch.onnx.export(model, dummy_input, "complete_multilevel_tempsal.onnx", export_params=True, opset_version=11, do_constant_folding=True)


# Export the model to ONNX format
torch.onnx.export(model, dummy_input, "model.onnx", 
                  export_params=True,        # Store the trained parameters
                  opset_version=11,          # ONNX version to export to
                  do_constant_folding=True,  # Optimize the model by folding constant nodes
                  input_names = ['input'],   # Input tensor name
                  output_names = ['output'], # Output tensor name
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
print("export completed")
