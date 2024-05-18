import os.path
import matplotlib.pyplot as plt
import explanations_emonet, global_analysis, local_analysis, emonet
import pandas as pd
import numpy as np
import csv
import torch
import seaborn as sns
import global_statistics
from explanations_emonet import ExplanationsEmonet

expl_emo = ExplanationsEmonet()
expl_emo.explanations_emonet('test_images/friends_parc.jpg', 'friend_parc', show_plot=True)