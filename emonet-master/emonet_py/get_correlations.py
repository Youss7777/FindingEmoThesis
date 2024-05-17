import os.path
import matplotlib.pyplot as plt
import explanations_emonet, global_analysis, local_analysis, emonet
import pandas as pd
import numpy as np
import csv
import torch
import seaborn as sns
import global_statistics
import torch


emonet_df = pd.read_csv('emonet_outputs')
yolo_df = pd.read_csv('yolo_outputs')

def get_obj_vec(obj_name):
    df2 = yolo_df[['dir_image_path', 'detected_object']].copy()
    df2[f'{obj_name}_presence'] = (df2['detected_object'] == obj_name).astype(int)
    df3 = df2.groupby(by=['dir_image_path'], sort=False, as_index=False).sum()
    vec_obj = df3[f'{obj_name}_presence'].apply(lambda x: 1 if x > 1 else x)
    print('Shape vec_obj: ', vec_obj.shape)
    return vec_obj

def get_emo_vec(emo_name):
    vec_emo = (emonet_df['emonet_emotion'] == emo_name).astype(int)
    print('Shape vec_emo: ', vec_emo.shape)
    return vec_emo

def obj_emo_correlation(obj_name, emo_name):
    vec_obj = get_obj_vec(obj_name)
    vec_emo = get_emo_vec(emo_name)
    return vec_obj.corr(vec_emo)


print(obj_emo_correlation('Human face', 'Amusement'))
print(obj_emo_correlation('Wheel', 'Amusement'))
