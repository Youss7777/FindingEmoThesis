import os.path
import matplotlib.pyplot as plt
import explanations_emonet, global_analysis, local_analysis, emonet
import pandas as pd
import numpy as np
import csv
import torch
import global_statistics
from explanations_emonet import ExplanationsEmonet
from local_analysis import LocalAnalysis

file_name = 'young_woman.jpg'  #
file_path = 'test_images/'+file_name

# not required if .npy already saved and you only want to do local analysis
expl_emo = ExplanationsEmonet()
max_emotion, max_prob, pred, arousal, valence = expl_emo.explanations_emonet(file_path, file_name, show_plot=False)

l_a = LocalAnalysis()
df = l_a.local_analysis(file_path, file_name, max_emotion=max_emotion, max_prob=max_prob,
                        explanation_method='gradcam', nb_objects=5, show_output=True)
plt.show()

