
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import os
import pretained_models as  ptmodels
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import decode_predictions, preprocess_input, InceptionV3
from keras.utils.data_utils import GeneratorEnqueuer
from config import Config as cfg
from tqdm import trange
import matplotlib.pyplot as plt
import tensorflow as tf
config = tf.ConfigProto(allow_soft_placement=True)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

config.gpu_options.allow_growth = True


def run_submission(data_names, add_dim, model_func, file_type, model_path):
    x = data_process.get_data_from_files(root_path='../data/test/')
    model = model_func(trainable=False)
    model.load_weights(model_path)
    print(model.summary())
    submission(model, x, fnames)
    
def submission(model, x, fnames,  file = '../data/submission.csv'):
    
    predict_y = model.predict(x, batch_size = 256, verbose=0)
    predictions = predictions + [0.] * (len(filenames) - len(predictions))
    
    pred_string = ["{} {:.4f} 0.1 0.1 0.9 0.9".format(valid_mappings[x][0], y_pred) if x is not None else "" for x, y_pred in zip(hit_idxs, predictions)]
    
    df_sub = pd.DataFrame({"ImageId": [xid.split(".")[0].split("/")[-1] for xid in fnames], 
                        "PredictionString": pred_string})
     
    
    print df_sub['PredictionString'].value_counts().sort_values()
    print "Submission samples:%d, file:%s"%(len(fnames), file)
    df_sub.to_csv(file, index=False)


   




if __name__ == "__main__":
   
#     test_InceptionNet_v3()