Download the model checkpoint from: https://drive.google.com/drive/folders/1W92oXYra_OPYkR1W56D80iDexWIR7f7Z?usp=sharing Follow the instructions on inference.ipynb. This notebook provides predictions on temporal and image saliency together.

Evaluation scripts: 

Codalab_saliency_evaluation_script_with_print.py
eval_on_validation_3_metrics.py

If KL metric is Nan, it can be a division by zero error. Please check if the prediction or ground truth is all zeros. 
