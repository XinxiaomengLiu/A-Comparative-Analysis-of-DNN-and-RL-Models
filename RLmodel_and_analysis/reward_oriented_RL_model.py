#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 21:22:55 2023

@author: xinxiaomengliu
"""

import numpy as np
import random

class MCIncrementalAgentConstant():
    def __init__(self, k, epsilon, alpha, beta, bandits_mean):
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.qvalues = bandits_mean
    
    # in each trail, the model assumes that the paticipants make their choices according to a softmax distribution based on the q-values they learned so far.
    def choose(self):
        qvalues_mean = np.mean(self.qvalues)
        qvalues_std = np.std(self.qvalues)
        if qvalues_std == 0:
            qvalues_normalized = self.qvalues
        else:
            qvalues_normalized = [(r - qvalues_mean) / qvalues_std for r in self.qvalues]
        prob = []
        for i in range(self.k):
            prob.append(np.exp(self.beta * qvalues_normalized[i]))
        denominator = sum(prob)
        prob = [x / (denominator) for x in prob]
        p = random.random()
        if p < self.epsilon:
            action = np.random.randint(self.k)
        else:
            action = np.random.choice(range(len(prob)), p = prob)
        return action, prob
    
    # update q-values
    def learn(self, reward, action): 
        self.qvalues[action] = self.qvalues[action] + self.alpha * (reward - self.qvalues[action])


def search_max_accuracy(df, df_idx, df_bandits_history, bandits_mean, param_grid):
    
    results = {}
    bandit1_history = df_bandits_history['reward_c1']
    bandit2_history = df_bandits_history['reward_c2']
    bandit3_history = df_bandits_history['reward_c3']
    bandit4_history = df_bandits_history['reward_c4']
    
    for epsilon in param_grid['epsilon']:
        for alpha in param_grid['alpha']:
            for beta in param_grid['beta']:
                print(epsilon, alpha, beta)
                # each row represents all obs's accuracy (either 0 or 1) in a specific step in the 150-step trail
                acc_matrix = [[] for i in range(150)]
                # get every user's accuracy
                for id in df_idx:  
                    agent = MCIncrementalAgentConstant(4, epsilon, alpha, beta, bandits_mean)
                    history_obs = df[df['id'] == id].reset_index(drop = True)
                    choice_obs = history_obs['choice']
                    # 150 trails
                    for i in range(150): 
                        choice, prob = agent.choose() # let the agent choose
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
                acc = []
                for row in acc_matrix:
                    if len(row) != 0:
                        acc.append(sum(row)/len(row))
                results[(epsilon, alpha, beta)] = np.mean(acc)
    best_params = max(results, key = results.get)
    return best_params