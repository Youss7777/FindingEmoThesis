import os.path
import matplotlib.pyplot as plt
import explanations_emonet, global_analysis, local_analysis, emonet
import pandas as pd
import numpy as np
import csv
import torch
import seaborn as sns
import global_statistics

# Sample DataFrame
gs = global_statistics.GlobalStatistics(obj_importance_thres=0.8, emo_conf_thres=0.6, obj_conf_thres=0,
                                        ann_ambiguity_thres=4)

emonet_df = gs.emonet_outputs
yolo_df = gs.yolo_outputs


def create_vec_emotion(row, *args):
    if row[args[0]] == args[1]:
        val = 1
    else:
        val = 0
    return val


vec_images = pd.DataFrame(columns=["dir_image_path"])
vec_images["dir_image_path"] = pd.DataFrame(yolo_df["dir_image_path"].unique())


vec_human_face = []
def create_vec_object_2():
    vec = []
    for i, row in vec_images.iterrows():
        if (yolo_df[['dir_image_path', 'detected_object']].values == [row['dir_image_path'], 'Wheel']).all(
                axis=1).any():
            vec.append(1)
        else:
            vec.append(0)
        print(vec)
    return vec


#vec_joy = emonet_df.apply(create_vec_emotion, args=("emonet_emotion", "Joy"), axis=1)
#vec_human_face = vec_images.apply(create_vec_object, axis=1)

create_vec_object_2()
