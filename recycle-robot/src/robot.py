import gymnasium
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class Robot(learning_rate):
    def __init__(self):

        self.actions = ['wait', 'search', 'recharge']

        self.battery_high = True 
        
        self.step = 0
        
        self.numeric_preferences = np.zeros(3)

        self.actions_record = []

        self.mean_reward = 0

        self.list_mean_reward = []

    def recharge(self):
        self.battery_high = True

    def discarge(self):
        self.battery_high = False

    def remember(self, action):
        self.actions_record.append[action]

    def choose(self):
        
        probabilities = np.array([
        np.exp(preference) for preference in self.numeric_preferences
        ]) / (np.exp(self.numeric_preferences).sum())
                
        action = np.random.choice(self.actions, p=probabilities)

        self.remember(action)

        return action
    
         