import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Plots:
    def __init__(self, heatmap_array:np.array, cum_total_rewards:list, cum_avg_rewards:np.array):
        """
        Args:
            heatmap_array (np.array): 
            cum_total_rewards (list): _description_
            cum_avg_rewards (np.array): _description_
        """
        self.heatmap_array = heatmap_array
        self.cum_total_rewards = cum_total_rewards
        self.cum_avg_rewards = cum_avg_rewards
    
    def plot_policy_heatmap(self):
        sns.heatmap(self.heatmap_array, 
                    annot=True,
                    xticklabels=["wait", "search", "recharge"],
                    yticklabels=["high", "low"],
                    )
        plt.xlabel("Actions")
        plt.ylabel("Battery level")
        plt.title("Policy Heatmap")
        plt.show()
    
    def plot_cum_total_rewards(self):
        n = len(self.cum_total_rewards[0])
        epochs = [i for i in range(1, n+1)]
        for array in self.cum_total_rewards:
            plt.plot(epochs, array, color="blue", alpha=0.3)
        plt.plot(epochs, self.cum_avg_rewards, color="orange", label="Average reward curve")
        plt.xlabel("Epochs")
        plt.xlim(1, n)
        plt.ylabel("Rewards")
        plt.grid()
        plt.legend()
        plt.title("Cumulative reward curve per epoch")
        plt.show()