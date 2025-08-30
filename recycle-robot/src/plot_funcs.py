import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from config import *

class Plots:
    # Method to plot the policy heatmap
    @staticmethod
    def plot_policy_heatmap(heatmap_array:np.array):
        sns.heatmap(heatmap_array, 
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
    @staticmethod
    def plot_cum_total_rewards_policies(epochs, list_rewards_best_policy, title, filename):
        n = epochs
        epochs = [i for i in range(1, n+1)]
        for array in list_rewards_best_policy:
            plt.plot(epochs, array[1], alpha=0.8, label=f"LR = {array[0]}")
        
        plt.xlabel("Epochs")
        plt.xlim(1, n)
        plt.ylabel("Rewards")
        plt.grid()
        plt.legend()
        plt.title(title)
        plt.savefig(os.path.join(FIGURES_FOLDER(), f"{filename}.png"))
        plt.close()
