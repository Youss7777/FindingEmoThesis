import pandas as pd
import emonet
import matplotlib.pyplot as plt
import json
import copy
import seaborn as sns
import scipy as sp
import association_metrics as am
import os
import torch
import numpy as np
import correlation_utils
import global_statistics

object_to_corr_emo_obj = ['Person', 'Human face', 'Human eye', 'Human nose', 'Human mouth', 'Human head', 'Clothing',
                          'Sports equipment', 'Footwear', 'Plant', 'Flower', 'Animal', 'Food', 'Drink', 'Furniture',
                          'Weapon', 'Wheel', 'Building', 'Personal Care', 'Vehicle', 'Tableware', 'Kitchenware',
                          'Pillow', 'Jeans']
object_to_corr_ann_obj = ['Person', 'Human face', 'Human eye', 'Human nose', 'Human mouth', 'Human head', 'Clothing',
                          'Sports equipment', 'Footwear', 'Plant', 'Flower', 'Tree', 'Animal', 'Food', 'Drink',
                          'Furniture', 'Weapon', 'Wheel', 'Building', 'Personal Care', 'Girl', 'Vehicle']
explanations_methods_list = {'gradcam': 'GradCAM', 'ablation': 'AblationCAM', 'lime': 'LimeCAM', 'lrp': 'LRPCAM', 'lift': 'LiftCAM'}


# concatenate vectors from each method into method_df
def get_method_df():
    method_df = pd.DataFrame()
    for method in explanations_methods_list:
        gs = global_statistics.GlobalStatistics(obj_importance_thres=0.3, emo_conf_thres=0.5, obj_conf_thres=0.1,
                              ann_ambiguity_thres=3, method=method)
        correlation_matrix = global_statistics.analysis_emo_obj(gs, emo_to_corr=emonet.EmoNet.EMOTIONS,
                                                                obj_to_corr=object_to_corr_emo_obj, method='dice')
        mask = np.tril(np.ones(correlation_matrix.shape), k=-1).astype(bool)
        lower_triangle = correlation_matrix.where(mask)
        flattened = pd.DataFrame(lower_triangle.values[mask])
        flattened.columns = [explanations_methods_list[method]]
        method_df = pd.concat([method_df, flattened], axis=1)
    return method_df


# plot meta correlation matrix
def plot_meta_corr_matrix(method_df):
    meta_corr_matrix = method_df.corr(method='spearman')
    mask = np.triu(np.ones_like(meta_corr_matrix))  # mask for triangular matrix only
    plt.figure(figsize=(9.5, 7))
    m = sns.heatmap(meta_corr_matrix, xticklabels=True, yticklabels=True, annot=True, cmap='coolwarm', mask=mask, annot_kws={"size": 15})
    x_labels = m.get_xticklabels()
    y_labels = m.get_yticklabels()
    m.set_xticklabels(x_labels, fontsize=15)
    m.set_yticklabels(y_labels, fontsize=15)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.title('Spearman correlation', fontsize=15)
    plt.tight_layout()


# p values matrix
def p_matrix(method_df):
    meta_corr_matrix = method_df.corr(method='spearman')
    method_df = method_df.dropna()
    p_values = np.zeros_like(meta_corr_matrix)
    for row in range(meta_corr_matrix.shape[0]):
        for col in range(meta_corr_matrix.shape[1]):
            if row != col:  # p-value calculation makes sense only for different columns
                _, p_values[row, col] = sp.stats.spearmanr(method_df.iloc[:, row], method_df.iloc[:, col])
            else:
                p_values[row, col] = np.nan  # diagonal can be set to NaN or some other value
    p_values_df = pd.DataFrame(p_values, index=meta_corr_matrix.index, columns=meta_corr_matrix.columns)
    return p_values_df

# plot p value matrix
def plot_p_matrix(method_df):
    p_values_df = p_matrix(method_df)
    mask = np.triu(np.ones_like(p_values_df))
    plt.figure(figsize=(9, 7))
    m = sns.heatmap(p_values_df, xticklabels=True, yticklabels=True, annot=True, mask=mask, annot_kws={"size": 15})
    x_labels = m.get_xticklabels()
    y_labels = m.get_yticklabels()
    m.set_xticklabels(x_labels, fontsize=15)
    m.set_yticklabels(y_labels, fontsize=15)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.title('Spearman significance', fontsize=15)
    plt.tight_layout()


method_df = get_method_df()
plot_meta_corr_matrix(method_df)
plt.show()
