#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 1 18:15:33 2023

@author: xinxiaomengliu
"""
import pandas as pd
import numpy as np
import seq_2_accuracy
import seq_2_results
import seq_3_accuracy
import seq_3_results
import seq_4_accuracy
import seq_4_results
from reward_oriented_RL_model import search_max_accuracy
from acc_similarity_plot import avg_accuracy, similarity, plot_structure_acc_similarity


# load data
df = pd.read_csv('/Users/alexandralau/Desktop/16project/DataAllSubjectsRewards.csv')

# separate data into 3 groups according to payoff_group
df2 = df.loc[df['payoff_group'] == 2].reset_index(drop = True)
df3 = df.loc[df['payoff_group'] == 3].reset_index(drop = True)
df4 = df.loc[df['payoff_group'] == 4].reset_index(drop = True)


# calculate the average payoff for every bandit in every payoff structure
df2_idx = df2['id'].unique().tolist()
df2_single_obs = df2[df2['id'] == df2_idx[0]]
# get the 150 trails reward history of every bandit
df2_bandits_history = df2_single_obs[['reward_c1', 'reward_c2', 'reward_c3', 'reward_c4']]
# get the mean reward of every bandit
df2_bandits_mean = [np.mean(list(np.mean(df2_bandits_history)))]*4
df2_test_idx = list(range(264, 330))



# calculate the average payoff for every bandit in every payoff structure
df3_idx = df3['id'].unique().tolist()
df3_single_obs = df3[df3['id'] == df3_idx[0]]
# get the 150 trails reward history of every bandit
df3_bandits_history = df3_single_obs[['reward_c1', 'reward_c2', 'reward_c3', 'reward_c4']]
# get the mean reward of every bandit
df3_bandits_mean = [np.mean(list(np.mean(df3_bandits_history)))]*4
df3_test_idx = list(range(579, 642))




# calculate the average payoff for every bandit in every payoff structure
df4_idx = df4['id'].unique().tolist()
df4_single_obs = df4[df4['id'] == df4_idx[0]]
# get the 150 trails reward history of every bandit
df4_bandits_history = df4_single_obs[['reward_c1', 'reward_c2', 'reward_c3', 'reward_c4']]
# get the mean reward of every bandit
df4_bandits_mean = [np.mean(list(np.mean(df4_bandits_history)))]*4
df4_test_idx = list(range(901, 966))





params_grid = {
    'epsilon': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'alpha': [0.1, 0.2, 0.3, 0.4, 0.5],
    'beta': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}
best_params_df2 = search_max_accuracy(df2, df2_idx, df2_bandits_history, df2_bandits_mean, params_grid)
# best_params_df2 = (0, 0.1, 5)
df2_test_acc, df2_test_result = avg_accuracy(df2, df2_test_idx, df2_bandits_history, df2_bandits_mean, best_params_df2)
seq_2_accuracy = seq_2_accuracy.seq_2_accuracy
seq_2_results = [ list(t) for t in seq_2_results.seq_2_results]
df2_similarity = similarity(df2_test_result, seq_2_results)
plot_structure_acc_similarity(df2_bandits_history, seq_2_accuracy, df2_test_acc, df2_similarity)





best_params_df3 = search_max_accuracy(df3, df3_idx, df3_bandits_history, df3_bandits_mean, params_grid)
# best_params_df3 = (0, 0.2, 4)
df3_test_acc, df3_test_result = avg_accuracy(df3, df3_test_idx, df3_bandits_history, df3_bandits_mean, best_params_df3)
seq_3_accuracy = seq_3_accuracy.seq_3_accuracy
seq_3_results = [ list(t) for t in seq_3_results.seq_3_results]
df3_similarity = similarity(df3_test_result, seq_3_results)
plot_structure_acc_similarity(df3_bandits_history, seq_3_accuracy, df3_test_acc, df3_similarity)




best_params_df4 = search_max_accuracy(df4, df4_idx, df4_bandits_history, df4_bandits_mean, params_grid)
# best_params_df3 = (0, 0.1, 3)
df4_test_acc, df4_test_result = avg_accuracy(df4, df4_test_idx, df4_bandits_history, df4_bandits_mean, best_params_df4)
seq_4_accuracy = seq_4_accuracy.seq_4_accuracy
seq_4_results = [ list(t) for t in seq_4_results.seq_4_results]
df4_similarity = similarity(df4_test_result, seq_4_results)
plot_structure_acc_similarity(df4_bandits_history, seq_4_accuracy, df4_test_acc, df4_similarity)






