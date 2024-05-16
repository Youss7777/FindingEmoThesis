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
def create_pres_vec(row, kwargs):
    if row[kwargs[0]] == kwargs[1]:
        val = 1
    else:
        val = 0
    return val

vec_human_face = emonet_df.apply(create_pres_vec, ["emonet_emotion", "Joy"], axis=1)


# Create dummy variables for detected objects and emotions
dummy_objects = pd.get_dummies(df['detected_object'])
dummy_emotions = pd.get_dummies(df['emonet_emotion'])

# Concatenate dummy variables with the original DataFrame
#df = pd.concat([df, dummy_objects, dummy_emotions], axis=1)
df2 = pd.concat([dummy_objects, dummy_emotions], axis=1)

# Calculate correlations between objects and emotions
correlations = df2.corr()

# Extract correlations between specific objects and specific emotions
objects = df["detected_object"].unique()  # List of objects
emotions = ['Joy']  # List of emotions

for obj in objects:
    for emo in emotions:
        correlation = correlations.loc[obj, emo]
        print(f"Correlation between {obj} and {emo}: {correlation}")


sns.heatmap(correlations, annot=True, cmap="coolwarm")
plt.tight_layout()
plt.show()
