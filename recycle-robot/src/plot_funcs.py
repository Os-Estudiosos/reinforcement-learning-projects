import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from config import *

class Plots:
    def __init__(self, heatmap_array:np.array, list_rewards_best_policy:tuple, list_all_rewards_learning:tuple):
        """
        Args:
            heatmap_array (np.array): 2x3 numpy array
            list_rewards_learning (tuple): Tuple in the form: (epochs: int, rewards_best_policy: list of np.array)
            list_all_rewards_learning (tuple): Tuple in the form: (epochs: int, all_rewards_learning: list of dictionaries in the form: [{"lam":float, "rewards":list of arrays}, {...}, ...])
        """
        self.heatmap_array = heatmap_array
        self.list_all_rewards_learning = list_all_rewards_learning
        self.list_rewards_best_policy = list_rewards_best_policy
        
    
    # Method to plot the policy heatmap
    def plot_policy_heatmap(self):
        sns.heatmap(self.heatmap_array, 
                    annot=True,
                    xticklabels=["wait", "search", "recharge"],
                    yticklabels=["high", "low"],
                    )
        plt.xlabel("Actions")
        plt.ylabel("Battery level")
        plt.title("Policy Heatmap")
        plt.savefig(os.path.join(FIGURES_FOLDER(), "heatmap_policy.png"))
        plt.close()
    
    # Method to plot the cumulative reward curves among the average cumulative reward curve for the best policy
    def plot_cum_total_rewards_best_policy(self):
        n = self.list_rewards_best_policy[0]
        epochs = [i for i in range(1, n+1)]
        for array in self.list_rewards_best_policy[1]:
            plt.plot(epochs, array, color="blue", alpha=0.3)
        cum_avg_rewards = np.sum(self.list_rewards_best_policy[1], axis=0)
        plt.plot(epochs, cum_avg_rewards, color="orange", label="Average reward curve")
        plt.xlabel("Epochs")
        plt.xlim(1, n)
        plt.ylabel("Rewards")
        plt.grid()
        plt.legend()
        plt.title("Cumulative best policy reward curve per epoch")
        plt.savefig(os.path.join(FIGURES_FOLDER(), "cum_total_rewards_best_policy_curve.png"))
        plt.close()
    
    # Method to plot the cumulative average rewards for each learning rate
    def plot_avg_learning_rewards(self):
        n = self.list_rewards_best_policy[0]
        epochs = [i for i in range(1, n+1)]
        m = len(self.list_all_rewards_learning[1])
        for i in range(m):
            lam = self.list_all_rewards_learning[1][i]["lam"]
            rewards = self.list_all_rewards_learning[1][i]["rewards"]
            avg_rewards = np.sum(rewards, axis=0) 
            plt.plot(epochs, avg_rewards, label=f"Average reward curve, λ={lam}")
        plt.xlabel("Epochs")
        plt.xlim(1, n)
        plt.ylabel("Average rewards")
        plt.grid()
        plt.legend()
        plt.title("Cumulative average learning rewards curve per epoch")
        plt.savefig(os.path.join(FIGURES_FOLDER(), "avg_learning_rewards.png"))
        plt.close()