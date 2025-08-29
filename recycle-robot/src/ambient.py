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
            "wait": .4,
            'recharge': 1,
            "discharge": -3
        }
    
    def process_reward(self, action: str):
        """Processes the reward that will be given to the agent based on the
        action passed as a parameter

        Args:
            action (str): Wich action the agant took. Can be "wait", "search" or "recharge"

        Returns:
            int: The numeric signal (reward)
        """
        # Checks if the agent's battery is high or low
        is_battery_high = self.agent.battery_high
        agent_state = 'high' if self.agent.battery_high else 'low'
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
                    self.agent.recharge()
                    return [ self.rewards["discharge"], agent_state ]
            if action == "recharge":
                self.agent.recharge()
        
        # And then return the reward based on the action
        return [ self.rewards[action], agent_state ]
    
    def run(self):
        """Runs the ambient program
        """
        for i in range(self.EPOCHS):    # For each epoch
            print()
            print("="*200)
            print(f"Estamos na época {i}")

            print(f"Preferências numéricas: {self.agent.numeric_preferences}")

            A = np.exp(self.agent.numeric_preferences)
            P = A.copy()
            P[0,:2] = A[0,:2]/A[0,:2].sum(keepdims=True)
            P[1] = A[1]/A[1].sum(keepdims=True)

            print(f"Política: {P}")

            action = self.agent.choose()    # The agent chooses an action

            # Processes the reward based on the passed action
            reward, agent_state = self.process_reward(action)
            
            # The agent will learn based on the reward received
            self.agent.learn(reward, agent_state)
