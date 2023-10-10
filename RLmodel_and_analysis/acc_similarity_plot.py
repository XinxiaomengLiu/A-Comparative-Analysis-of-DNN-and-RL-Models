#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 1 20:30:57 2023

@author: xinxiaomengliu
"""
import numpy as np
import matplotlib.pyplot as plt 
from reward_oriented_RL_model import MCIncrementalAgentConstant

def avg_accuracy(df, df_idx, df_bandits_history, bandits_mean, params):
    # plot the result on test set
    epsilon, alpha, beta = params
    
    bandit1_history = df_bandits_history['reward_c1']
    bandit2_history = df_bandits_history['reward_c2']
    bandit3_history = df_bandits_history['reward_c3']
    bandit4_history = df_bandits_history['reward_c4']
    result = [[] for i in range(150)]
    acc_matrix = [[] for i in range(150)]
    # get every user's accuracy
    for id in df_idx:  
        agent = MCIncrementalAgentConstant(4, epsilon, alpha, beta, bandits_mean)
        history_obs = df[df['id'] == id].reset_index(drop = True)
        choice_obs = history_obs['choice']
        # 150 trails
        for i in range(150): 
            choice, prob = agent.choose() # let the agent choose
            result[i].append(choice)
            if choice == 0:
                reward = bandit1_history[i]
            elif choice == 1:
                reward = bandit2_history[i]
            elif choice == 2:
                reward = bandit3_history[i]
            elif choice == 3:
                reward = bandit4_history[i]
            agent.learn(reward, choice)
            if np.isnan(choice_obs[i]):
                pass
            elif choice == (int(choice_obs[i])-1):
                acc_matrix[i].append(1)
            else:
                acc_matrix[i].append(0)
    
    # calculate average accuracy in each step, which is, the average of each row
    acc = []
    for row in acc_matrix:
        if len(row) != 0:
            acc.append(sum(row)/len(row))
    return acc, result

def similarity(df_test_result, seq_results):
    df_similarity_matrix = [[] for i in range(150)]
    for i in range(150):
        for j in range(len(df_test_result[0])):
            max_len_NN = len(seq_results[j])
            if i <  max_len_NN:
                RO = df_test_result[i][j]
                NN = seq_results[j][i]
                if RO == NN:
                    df_similarity_matrix[i].append(1)
                else:
                    df_similarity_matrix[i].append(0)
    df_similarity = []
    for row in df_similarity_matrix:
        if len(row) != 0:
            df_similarity.append(sum(row)/len(row))
    return df_similarity


def plot_structure_acc_similarity(df_bandits_history, seq_accuracy, df_test_acc, df_similarity):
    
    fig, axes = plt.subplots(nrows = 3, ncols = 1, figsize = (10, 9))

    df_bandit1_history = list(df_bandits_history['reward_c1'])
    df_bandit1_history = list(df_bandits_history['reward_c1'])
    df_bandit2_history = list(df_bandits_history['reward_c2'])
    df_bandit3_history = list(df_bandits_history['reward_c3'])
    df_bandit4_history = list(df_bandits_history['reward_c4'])

    axes[0].plot(df_bandit1_history, color = 'blue')
    axes[0].plot(df_bandit2_history, color = 'orange')
    axes[0].plot(df_bandit3_history, color = 'green')
    axes[0].plot(df_bandit4_history, color = 'red')
    axes[0].set_title('Payoff Structure 4')
    axes[0].set_xlabel('Trial')
    axes[0].set_ylabel('Reward')
    axes[0].legend(['reward_c1', 'reward_c2', 'reward_c3', 'reward_c4'])



    axes[1].plot(seq_accuracy, color = 'purple')
    axes[1].plot(df_test_acc, color = 'teal')
    axes[1].set_title("Models' Accuracy")
    axes[1].set_xlabel('Trial')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend(['Neural Network Model', 'Reward-Oriented Model'])

    axes[2].plot(df_similarity, color = 'blue')
    axes[2].set_title("Models' Similarity")
    axes[2].set_xlabel('Trial')
    axes[2].set_ylabel('Similarity')

    fig.subplots_adjust(hspace=0.5, bottom=0.1, top=0.9)
