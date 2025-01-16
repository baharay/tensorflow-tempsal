import os
import numpy as np
from skimage import io, transform
import numpy as np
# from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, entropy
import sys
import glob
from distutils.dir_util import copy_tree

def normalize_map(s_map):
    # normalize the salience map
    batch_size = 1
    w = 512
    h = 512

    s_map = s_map.reshape(batch_size, w, h)
    min_s_map = np.min(s_map, axis=(1, 2)).reshape(batch_size, 1, 1)
    max_s_map = np.max(s_map, axis=(1, 2)).reshape(batch_size, 1, 1)

    norm_s_map = (s_map - min_s_map) / (max_s_map - min_s_map)
    return norm_s_map

def auc_judd(saliencyMap, fixationMap, jitter=True, normalize=True):
    # Ensure saliencyMap and fixationMap have the same shape
    # if saliencyMap.shape != fixationMap.shape:
    #     saliencyMap = resize_map(saliencyMap, fixationMap.shape)

    if normalize:
        saliencyMap = normalize_map(saliencyMap)
        # saliencyMap = normalize_map(fixationMap)

    if not fixationMap.any():
        print('Error: no fixationMap')
        return float('nan')

    # Jitter saliency map slightly to disrupt ties of the same numbers
    if jitter:
        saliencyMap += np.random.random(saliencyMap.shape) / 10**7

    # Normalize saliency map
    saliencyMap = (saliencyMap - saliencyMap.min()) / (saliencyMap.max() - saliencyMap.min())

    if np.isnan(saliencyMap).all():
        print('NaN saliencyMap')
        return float('nan')

    S = saliencyMap.flatten()
    F = fixationMap.flatten()

    Sth = S[F > 0]  # Saliency map values at fixation locations
    Nfixations = len(Sth)
    Npixels = len(S)

    allthreshes = sorted(Sth, reverse=True)  # Sort saliency map values to sweep through values
    tp = np.zeros((Nfixations + 2))
    fp = np.zeros((Nfixations + 2))
    tp[0], tp[-1] = 0, 1
    fp[0], fp[-1] = 0, 1

    for i in range(Nfixations):
        thresh = allthreshes[i]
        aboveth = (S >= thresh).sum()  # Total number of saliency map values above threshold
        tp[i + 1] = float(i + 1) / Nfixations  # Ratio saliency map values at fixation locations above threshold
        over = (Npixels - Nfixations)
        if  over== 0:
            over = 1

        fp[i + 1] = float(aboveth - i) / over  # Ratio other saliency map values above threshold

    score = np.trapz(tp, x=fp)
    return score
def similarity(s_map, gt):
    batch_size = 1
    w = 512
    h = 512

    s_map = normalize_map(s_map)
    gt = normalize_map(gt)

    sum_s_map = np.sum(s_map, axis=(1, 2)).reshape(batch_size, 1, 1)
    sum_gt = np.sum(gt, axis=(1, 2)).reshape(batch_size, 1, 1)

    s_map = s_map / sum_s_map
    gt = gt / sum_gt

    s_map_flat = s_map.reshape(batch_size, -1)
    gt_flat = gt.reshape(batch_size, -1)

    similarity_score = np.mean(np.sum(np.minimum(s_map_flat, gt_flat), axis=1))
    return similarity_score
import numpy as np

def my_cc(s_map, gt):
    batch_size = 1
    w = 512
    h = 512

    # Calculate the mean and std for s_map
    mean_s_map = np.mean(s_map.reshape(batch_size, -1), axis=1).reshape(batch_size, 1, 1)
    std_s_map = np.std(s_map.reshape(batch_size, -1), axis=1).reshape(batch_size, 1, 1)

    # Calculate the mean and std for gt
    mean_gt = np.mean(gt.reshape(batch_size, -1), axis=1).reshape(batch_size, 1, 1)
    std_gt = np.std(gt.reshape(batch_size, -1), axis=1).reshape(batch_size, 1, 1)

    # Normalize the saliency map and ground truth
    s_map = (s_map - mean_s_map) / (std_s_map + 1e-7)  # Added a small value to avoid division by zero
    gt = (gt - mean_gt) / (std_gt + 1e-7)

    # Calculate the components for the correlation coefficient
    ab = np.sum((s_map * gt).reshape(batch_size, -1), axis=1)
    aa = np.sum((s_map * s_map).reshape(batch_size, -1), axis=1)
    bb = np.sum((gt * gt).reshape(batch_size, -1), axis=1)

    # Calculate and return the mean correlation coefficient
    return np.mean(ab / (np.sqrt(aa * bb) + 1e-7))  # Added a small value to avoid division by zero


def calculate_auc(pred, gt):
    # pred = normalize_map(pred)
    # gt = normalize_map(gt)   
    pred = pred.flatten()
    gt = gt.flatten()
    auc = auc_judd( pred,gt)
    return auc

def calculate_nss(pred, gt):
    pred_mean = np.mean(pred)
    pred_std = np.std(pred)
    if pred_std == 0:
        return 0
    pred_normalized = (pred - pred_mean) / pred_std
    nss = np.mean(pred_normalized * gt)
    return nss

def calculate_cc(pred, gt):
  
    # pred = pred.flatten()
    # gt = gt.flatten()
    cc  = my_cc(pred, gt)
    return cc

def calculate_kld(pred, gt):
    epsilon = 1e-10  # Small constant to avoid log(0)

    pred = pred / np.sum(pred)
    gt = gt / np.sum(gt)
    pred = np.clip(pred, epsilon, None)

    kld = entropy(gt, pred)
    return kld


def load_images_from_folder(folder, target_size=(512, 512)):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img = io.imread(os.path.join(folder, filename), as_gray=True)
            if img is not None:
                img_resized = transform.resize(img, target_size, anti_aliasing=True)
                images.append(img_resized)
    return images

def evaluate_model(predictions_folder, gt_folder):

    # print("predictions:")
    # for thing in os.listdir(predictions_folder):
    #     print("predictions input:", thing)
    predictions = load_images_from_folder(predictions_folder)
    # print("ground_truths:")
    # for thing in os.listdir(gt_folder):
    #     print("ground_truths input:", thing)
    ground_truths = load_images_from_folder(gt_folder)
    # print(len(predictions))
    # print(len(ground_truths))

    assert len(predictions) == len(ground_truths), "Number of prediction and ground truth images must be the same."

    auc_scores = []
    sim_scores = []
    cc_scores = []
    kld_scores = []

    for pred, gt in zip(predictions, ground_truths):
        auc_scores.append(calculate_auc(pred, gt))
       # print('AUC', np.mean(auc_scores))
        sim_scores.append(similarity(pred, gt))
       # print('SIM', np.mean(sim_scores))
        cc_scores.append(calculate_cc(pred, gt))
       # print('CC', np.mean(cc_scores))
        kld_scores.append(calculate_kld(pred, gt))
       # print('KLD', np.mean(kld_scores))

    metrics = {
        'AUC': np.mean(auc_scores),
        'SIM': np.mean(sim_scores),
        'CC': np.mean(cc_scores),
        'KLD': np.mean(kld_scores)
    }

    return metrics



# def print_directory_structure(directory, indent_level=0):
#     """
#     Recursively prints the directory structure.
    
#     Args:
#     directory (str): The directory to start printing the structure from.
#     indent_level (int): The level of indentation (used for recursion).
#     """
#     try:
#         items = os.listdir(directory)
#     except PermissionError:
#         print(" " * indent_level + "[Permission Denied]")
#         return
    
#     for item in items:
#         item_path = os.path.join(directory, item)
#         if os.path.isdir(item_path):
#             print(" " * indent_level + f"Directory: {item}")
#             print_directory_structure(item_path, indent_level + 2)
#         else:
#             print(" " * indent_level + f"File: {item}")




if __name__ == "__main__":
    print("I'm a NEW submission!")


    program_directory = sys.argv[1]
    input_directory = sys.argv[2]
    output_directory = sys.argv[3]



    base_directory = "."
    # print(f"Directory structure for {base_directory}:\n")
    # print_directory_structure(base_directory)



    # hidden_directory = sys.argv[4]
    # shared_directory = sys.argv[5]
    # submission_directory = sys.argv[6]
    # ingestion_directory = sys.argv[7]


    # print("Sharing the code submitted...")  
    # copy_tree (program_directory, shared_directory)

    # print("Write some stuff to shared file...") 
    # for i in range(10):
    #     name = "{}_{}".format("newfile", i)
    #     open(os.path.join(shared_directory, name), 'w+').write('test!')
        
    # print("Writing answer to...", output_directory)

    # # answer_path = os.path.join(output_directory, "answer.txt")
    # # open(answer_path, 'w+').write('Hello World!')
        
    # print("Program directory:", program_directory)
    # for thing in glob.glob(os.path.join(program_directory, '*')):
    #     print("program:", thing)
        
    # print("Input directory:", input_directory)
    # for thing in glob.glob(os.path.join(input_directory, '*')):
    #     print("input:", thing)
        
    # print("Output directory:", output_directory)
    # for thing in glob.glob(os.path.join(output_directory, '*')):
    #     print("output:", thing)
        
    # # print("Hidden directory:", hidden_directory)
    # for thing in glob.glob(os.path.join(hidden_directory, '*')):
    #     print("hidden:", thing)
        
    # print("Shared directory:", shared_directory)
    # for thing in glob.glob(os.path.join(shared_directory, '*')):
    #     print("shared:", thing)
        
    # print("Submission directory:", submission_directory)
    # for thing in glob.glob(os.path.join(submission_directory, '*')):
    #     print("submission:", thing)
        
    # print("Ingestion directory:", ingestion_directory)
    # for thing in glob.glob(os.path.join(ingestion_directory, '*')):
    #     print("ingestion:", thing)
        



    predictions_folder = "./input/res/" # Evaluate the model on C:/Users/devcloud/bahar_projects/git-tf/tensorflow-tempsal/tf-test/dataset/ExtractedFolder/salicon/images/val/
    gt_folder = "./input/ref/" #C:/Users/devcloud/bahar_projects/git-tf/tensorflow-tempsal/tf-test/dataset/ExtractedFolder/salicon/maps/val/

    metrics = evaluate_model(predictions_folder, gt_folder)
    print(f"Model Evaluation Metrics:\nAUC: {metrics['AUC']}\nSIM: {metrics['SIM']}\nCC: {metrics['CC']}\nKL: {metrics['KLD']}")

     # Save metrics to a file
    scores_path = os.path.join(output_directory, "scores.txt")
    with open(scores_path, 'w') as f:
        f.write(f"AUC: {metrics['AUC']}\n")
        f.write(f"CC: {metrics['CC']}\n")
        f.write(f"KL: {metrics['KLD']}\n")
        f.write(f"SIM: {metrics['SIM']}\n")
    print(f"Scores saved to {scores_path}")
