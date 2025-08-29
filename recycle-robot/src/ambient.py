import gymnasium
from src.robot import Robot


class Ambient:
    def __init__(self, alpha, beta, learning_rate, epochs):
        self.alpha = alpha
        self.beta = beta
        self.learning_rate = learning_rate

        self.agent = Robot(learning_rate)

        self.EPOCHS = epochs
    
    def process_reward(self, action):
        ...
    
    def run(self):
        for _ in range(self.EPOCHS):
            action = self.agent.choose()

            reward = self.process_reward(action)
