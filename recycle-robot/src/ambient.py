import gymnasium
import numpy as np
from src.robot import Robot


class Ambient:
    def __init__(self, alpha, beta, learning_rate, epochs):
        self.alpha = alpha
        self.beta = beta
        self.learning_rate = learning_rate

        self.agent = Robot(learning_rate)

        self.EPOCHS = epochs

        self.rewards = {
            "search": 1,
            "wait": abs(np.random.standard_normal()),
            'recharge': 0,
            "discharge": -3
        }
    
    def process_reward(self, action: str) -> int:
        """Processes the reward that will be given to the agent based on the
        action passed as a parameter

        Args:
            action (str): Wich action the agant took. Can be "wait", "search" or "recharge"

        Returns:
            int: The numeric signal (reward)
        """
        # Checks if the agent's battery is high or low
        is_battery_high = self.agent.battery_high

        # If it has a high battery
        if is_battery_high:
            if action == "search":  # I check wich action it tooked
                # I randomize with probability 1-alpha that the battery changed its state
                # from high to low battery
                battery_got_low = np.random.random() < (1 - self.alpha)

                if battery_got_low:  # If it got low
                    self.agent.discharge()  # I execute the discharge function
        else:
            if action == "search":
                ran_out_of_battery = np.random.random() < (1 - self.beta)

                if ran_out_of_battery:
                    return self.rewards["discharge"]
        
        # And then return the reward based on the action
        return self.rewards[action]
    
    def run(self):
        """Runs the ambient program
        """
        for _ in range(self.EPOCHS):    # For each epoch
            action = self.agent.choose()    # The agent chooses an action

            # Processes the reward based on the passed action
            reward = self.process_reward(action)
            
            # The agent will learn based on the reward received
            self.agent.learn(reward)
