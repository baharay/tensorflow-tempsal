import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import os


# Function to save individual images
def save_image(image, filepath, temporal_map=True):
    """Save a tensor image to the specified filepath."""
    # Remove batch dimension if it exists
    image = tf.squeeze(image, axis=0) if image.shape[0] == 1 else image
    cmap = "hot"
    if not temporal_map:
        cmap='gray'
    plt.imsave(filepath, image, cmap=cmap)  # Save as grayscale or specify color



# Equivalent to PyTorch's `to_np`
def to_np(tensor):
    return tensor.numpy()

def plot_side_by_side(img_array, names=[], colormap="gray"):
    """Display multiple images side by side with optional titles."""
    num_imgs = len(img_array)
    plt.figure(figsize=(15, 5))
    for i in range(num_imgs):
        plt.subplot(1, num_imgs, i + 1)

        # Remove the first dimension if necessary (for grayscale images)
        img = tf.squeeze(img_array[i], axis=0).numpy() if img_array[i].shape[0] == 1 else img_array[i]
        
        plt.imshow(img, cmap=colormap)
        if len(names) > 0:
            plt.title(str(names[i]))
        plt.axis("off")
    plt.show()

# Image transformations for input images and ground truth
def img_transform(image):
    """Resize and normalize an image for model input."""
    image = image.resize((256, 256))
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    return tf.transpose(image, [2, 0, 1])  # Move channels to the first dimension

def gt_transform(image):
    """Resize and prepare ground truth image."""
    image = image.resize((256, 256))
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    return tf.convert_to_tensor(image, dtype=tf.float32)

# Functions to load images
def get_image(img_path):
    """Load and transform an RGB image."""
    img = Image.open(img_path).convert('RGB')
    img = img_transform(img)
    return tf.expand_dims(img, axis=0)  # Add batch dimension

def get_image_nonorm(img_path):
    """Load and transform an RGB image without normalization."""
    img = Image.open(img_path).convert('RGB')
    img = gt_transform(img)
    return tf.expand_dims(img, axis=0)

def get_gt_tensor(img_path):
    """Load and transform ground truth grayscale image."""
    img = Image.open(img_path).convert('L')
    img = gt_transform(img)
    return tf.expand_dims(img, axis=0)

def predict(image):
    """This function should perform the actual prediction."""
    output = model(input=image)
    temp_pred = output["onnx::Concat_4869"]
    fin_pred = output["output"]
    return fin_pred, temp_pred


predictions_dir = './testing/predictions/'
# Provide the paths to the images you want to test
image_name = "COCO_val2014_000000000208"
image_paths = ["./testing/images/" + image_name + ".jpg"]
gt_paths = ["./testing/gt/" + image_name + ".png"]
gt_vol_paths = [
    "./testing/gt/" + image_name + "_0.png",
    "./testing/gt/" + image_name + "_1.png",
    "./testing/gt/" + image_name + "_2.png",
    "./testing/gt/" + image_name + "_3.png",
    "./testing/gt/" + image_name + "_4.png"
]
# model = tf.keras.models.load_model('./model_tf')
# for layer in model.layers:
#     print(layer)
#     layer.trainable = False  # Freezing all layers
model = tf.saved_model.load("model_tf")
# Check available signatures
print(model.signatures)  # Lists available methods/signatures for inference

# Loop over the images, make predictions, and display the results
for image_path in image_paths:
    # Read image
    image = get_image(image_path)
    
    # Predict saliency (assuming model returns a final prediction and temporal predictions)
    fin_pred, temp_pred = predict(image)  # The image already has the correct shape
    # print("temp_pred type:", type(temp_pred))  # Check the type of temp_pred
    temp_pred = tf.squeeze(temp_pred, axis=0)
    fin_pred = tf.squeeze(fin_pred, axis=0)

    # Ensure temp_pred is a tensor before calling .numpy()
    if isinstance(temp_pred, tf.Tensor):
        temp_preds = [to_np(temp_pred[i]) for i in range(5)]
    else:
        print("temp_pred is not a tensor. Check predict() function.")
    # Ground truth tensors
    gt_tensors = [to_np(get_gt_tensor(gt_vol_paths[i])) for i in range(5)]

    # Temporal predictions
    temp_preds = [to_np(temp_pred[i]) for i in range(5)]
    plot_side_by_side(gt_tensors, 
                      ["0-1 s Ground Truth", "1-2 s Ground Truth", "2-3 s Ground Truth", "3-4 s Ground Truth", "4-5 s Ground Truth"], 
                      colormap="hot")
    
    plot_side_by_side(temp_preds, 
                      ["0-1 s Prediction", "1-2 s Prediction", "2-3 s Prediction", "3-4 s Prediction", "4-5 s Prediction"], 
                      colormap="hot")

    # Display input image, ground truth, and final prediction
    plot_side_by_side([to_np(get_image_nonorm(image_path)), to_np(get_gt_tensor(gt_paths[0])), to_np(fin_pred)], 
                      ["Input image", "0-5 s Ground Truth", "0-5 s Prediction"], 
                      colormap="gray")

    # After making predictions, save each prediction one by one
    for i, pred in enumerate(temp_preds):
        # Save each prediction with a unique filename
        save_image(pred, os.path.join(predictions_dir, f'temporal_saliency_prediction_{i}.png'))

    # Example for final prediction
    save_image(fin_pred, os.path.join(predictions_dir, 'image_saliency_prediction.png'),temporal_map=False)