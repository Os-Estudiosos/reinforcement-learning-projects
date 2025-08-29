import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class Robot:
    def __init__(self, learning_rate):

        self.learning_rate = learning_rate              #the parameter alpha to update the preferences

        self.actions = ['wait', 'search', 'recharge']   #list of possible actions 

        self.battery_high = True                        #control of the state battery
        
        self.step = 0                                   #quantity of step
        
        self.numeric_preferences = np.zeros((2,3))      #the preferences of the robot

        self.actions_record = []                        #list of all actions(to plot)

        self.mean_reward = 0                            #compute the mean reward to update the preference

        self.culmutative_reward = 0

        self.reward_record = []

        self.list_mean_reward = []                      #list of the mean rewards(to plot)

        self.atual_action = 'None'                      #save the atual action          

    def recharge(self):
        """Change the state of the robot to high batterry
        """
        self.battery_high = True

    def discharge(self):
        """Change the state of the robot to low batterry
        """
        self.battery_high = False

    def save_action(self, action: str):
        """Make a list to save the actions did.

        Args:
            action (str): name of the action.
        """
        self.actions_record.append(action)

    def probabilities_comp(self, preferences: np.array) -> np.array :
        """Compute the array of probabilities of selection each action

        Args:
            preferences (array): array of the preferences of the robot of each action. 

        Returns:
            array: an array of the probability of selection each action
        """
        prob = np.array([                                        
            np.exp(preference) for preference in preferences    #for each preference expoents the value
        ]) / (np.exp(preferences).sum())                        #and divide by the sum of total expoents of each value

        return prob

    def choose(self) -> str :
        """Choose the next action based in the preferences

        Returns:
            str: action choosed 
        """
        if self.battery_high == True:                                                   #if the state is high battery
            probabilities = self.probabilities_comp(self.numeric_preferences[0][:2])    #compute each probability (just take the actions 'search' and 'wait' because recharge its not an option)
            self.atual_action = np.random.choice(self.actions[:2], p=probabilities)     #choose on randomly based on the probability    
        else:                                                                           #else the state is low battery 
            probabilities = self.probabilities_comp(self.numeric_preferences[1])        #same thing but taking all the three states
            self.atual_action = np.random.choice(self.actions, p=probabilities)

        self.save_action(self.atual_action)                                             #append on the list of actions 

        self.step += 1                                                                  #increments the steps

        return self.atual_action
    
    def reset_rewards(self):
        self.culmutative_reward = 0
        self.reward_record.clear()

    def learn(self, reward, previous_state):
        self.culmutative_reward += reward
        self.reward_record.append(self.culmutative_reward)

        if previous_state == 'high':
            probabilities = self.probabilities_comp(self.numeric_preferences[0][:2])
            if self.atual_action == 'wait':
                self.numeric_preferences[0][0] = self.numeric_preferences[0][0] + self.learning_rate * (reward - self.mean_reward) * (1 - probabilities[0])
                self.numeric_preferences[0][1] = self.numeric_preferences[0][1] - self.learning_rate * (reward - self.mean_reward) * probabilities[1]
            elif self.atual_action == 'search':
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
        

        
         