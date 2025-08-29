import numpy as np
from src.ambient import Ambient
from src.robot import Robot
from src.plot_funcs import Plots
from yaspin import yaspin
import threading
from utils import *


def main():
    EPOCHS = 1000
    ALPHA = .5
    BETA = .5
    lambda_values = [
        0.001,
        0.025,
        0.05,
        0.01,
        0.06,
        0.1
    ]

    agents: list[Robot] = []
    
    # Visual part
    clear_terminal()

    print("-"*50)
    print("\033[92;5mRECYCLE BIN ROBOT\033[m")
    print("-"*50)

    print('\033[91mAMBIENT VARIABLES\033[m')
    print(f'alpha := Probability of staying in high battery after search for can = {ALPHA}')
    print(f'beta := Probability of staying in low battery after search for can = {BETA}')
    print(f'number of epochs = {EPOCHS}')

    print("-"*20)

    for lr in lambda_values:
        ambient = Ambient(
            ALPHA,
            BETA,
            lr,
            EPOCHS
        )

        with yaspin(text=f"Training robot with Learning Rate {lr}", color="cyan") as spinner:
            t = threading.Thread(target=ambient.run)
            t.start()
            t.join()
            spinner.ok("✔️")

        agents.append(ambient.agent)
    
    # Choosing the best agent
    best_agent = max(agents, key=lambda agent: agent.mean_reward)

    print('-'*20)
    print('\033[91mBEST AGENT\033[m')
    print(f'Learning Rate: {best_agent.learning_rate}')
    print('Numeric Preferences:')
    print(best_agent.numeric_preferences)

    print('-'*20)
    A = np.exp(best_agent.numeric_preferences)
    P = A.copy()
    P[0,:2] = A[0,:2]/A[0,:2].sum()
    P[1] = A[1] / A[1].sum()
    P[0,2] = 0

    plots = Plots(
        P,
        (
            
        )
    )

    # with yaspin(text=f"Training robot with Learning Rate {lr}", color="cyan") as spinner:
    #     t = threading.Thread(target=ambient.run)
    #     t.start()
    #     t.join()
    #     spinner.ok("✔️")

    print('-'*50)


if __name__ == "__main__":
    main()
