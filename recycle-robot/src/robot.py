import gymnasium
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class Robot:
    def __init__(self, learning_rate):

        self.learning_rate = learning_rate

        self.actions = ['wait', 'search', 'recharge']

        self.battery_high = True 
        
        self.step = 0
        
        self.numeric_preferences = np.zeros((2,3))

        self.actions_record = []

        self.mean_reward = 0

        self.list_mean_reward = []

        self.atual_action = ''

    def recharge(self):
        self.battery_high = True

    def discarge(self):
        self.battery_high = False

    def remember(self, action):
        self.actions_record.append[action]

    def probabilities_comp(self, preferences):
        prob = np.array([
                np.exp(preference) for preference in preferences
        ]) / (np.exp(preferences).sum())

        return prob

    def choose(self):
        
        if self.battery_high == True:    
            probabilities = self.probabilities_comp(self.numeric_preferences[0][:2])
        else:
            probabilities = self.probabilities_comp(self.numeric_preferences[1])
                    
        self.atual_action = np.random.choice(self.actions, p=probabilities)

        self.remember(self.atual_action)

        self.step += 1

        return self.atual_action
    
    
    def learn(self, reward):

        if self.battery_high:    
            probabilities = self.probabilities_comp(self.numeric_preferences[0][:2])
            if self.atual_action == 'wait':
                self.numeric_preferences[0][0] = self.numeric_preferences[0][0] + self.learning_rate * (reward - self.mean_reward) * (1 - probabilities[0])
                self.numeric_preferences[0][1] = self.numeric_preferences[0][1] - self.learning_rate * (reward - self.mean_reward) * probabilities[1]
            else:
                self.numeric_preferences[0][1] = self.numeric_preferences[0][1] + self.learning_rate * (reward - self.mean_reward) * (1 - probabilities[1])
                self.numeric_preferences[0][0] = self.numeric_preferences[0][0] - self.learning_rate * (reward - self.mean_reward) * probabilities[0]
                
        else:
            probabilities = self.probabilities_comp(self.numeric_preferences[1])
            if self.atual_action == 'wait':
                self.numeric_preferences[1][0] = self.numeric_preferences[1][0] + self.learning_rate * (reward - self.mean_reward) * (1 - probabilities[0])
                self.numeric_preferences[1][1] = self.numeric_preferences[1][1] - self.learning_rate * (reward - self.mean_reward) * probabilities[1]
                self.numeric_preferences[1][2] = self.numeric_preferences[1][2] - self.learning_rate * (reward - self.mean_reward) * probabilities[2]
            elif self.atual_action == 'search':
                self.numeric_preferences[1][0] = self.numeric_preferences[1][0] - self.learning_rate * (reward - self.mean_reward) * probabilities[0]
                self.numeric_preferences[1][1] = self.numeric_preferences[1][1] + self.learning_rate * (reward - self.mean_reward) * (1 - probabilities[1])
                self.numeric_preferences[1][2] = self.numeric_preferences[1][2] - self.learning_rate * (reward - self.mean_reward) * probabilities[2]
            else:
                self.numeric_preferences[1][0] = self.numeric_preferences[1][0] - self.learning_rate * (reward - self.mean_reward) * probabilities[0]
                self.numeric_preferences[1][1] = self.numeric_preferences[1][1] - self.learning_rate * (reward - self.mean_reward) * probabilities[1]
                self.numeric_preferences[1][2] = self.numeric_preferences[1][2] + self.learning_rate * (reward - self.mean_reward) * (1 - probabilities[2])
         
        self.mean_reward = 1/self.step * (reward + (self.step - 1) * self.mean_reward)

        self.list_mean_reward.append(self.mean_reward)
        

        
         