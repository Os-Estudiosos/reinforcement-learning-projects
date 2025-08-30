import numpy as np
from src.ambient import Ambient
from src.robot import Robot
from src.plot_funcs import Plots
from yaspin import yaspin
import threading
from utils import *


def main():
    EPOCHS = 1000
    ALPHA = .6
    BETA = .4
    lambda_values = [
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35
    ]

    agents: list[Robot] = []
    
    # Visual part
    extra_message = ""
    while True:
        clear_terminal()

        print("-"*50)
        print("\033[92;5mRECYCLE BIN ROBOT\033[m")
        print("-"*50)

        print('\033[96mAMBIENT VARIABLES\033[m')
        print(f'alpha := Probability of staying in high battery after search for can = {ALPHA}')
        print(f'beta := Probability of staying in low battery after search for can = {BETA}')
        print(f'number of epochs = {EPOCHS}')

        print(f'\033[91m{extra_message}\033[m')

        resp = input('\033[97mWould you like to digit your the values of alpha and beta? (Y/N): \033[m').strip().upper()[0]

        if resp == 'N':
            break
        elif resp == 'Y':
            try:
                ALPHA = float(input('\033[97mType the alpha value: \033[m'))
                BETA = float(input('\033[97mType the beta value: \033[m'))
                EPOCHS = int(input('\033[97mType the number of epochs: \033[m'))

                if not (0 < ALPHA < 1 or 0 < BETA < 1):
                    raise ValueError
            except ValueError:
                extra_message = "You didn't type a number or you didn't type a number between 0 and 1, try again, please..."
            else:
                break
        else:
            extra_message = "Please, type Y or N"

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

    A = np.exp(best_agent.numeric_preferences)
    P = A.copy()
    P[0,:2] = A[0,:2]/A[0,:2].sum()
    P[1] = A[1] / A[1].sum()
    P[0,2] = 0

    print("Best policy:")
    print(P)

    print('-'*20)

    # Best Policy Heatmap
    Plots.plot_policy_heatmap(P)

    # Best agent culmutative rewards
    Plots.plot_cum_total_rewards_policies(
        EPOCHS,
        [
            [ best_agent.learning_rate, best_agent.reward_record ]
        ],
        "BEST AGENT'S CULMUTATIVE REWARD OVER THE EPOCHS",
        "best_agent_culmutative_reward_over_the_epochs"
    )

    # All agents culmutative rewards
    Plots.plot_cum_total_rewards_policies(
        EPOCHS,
        [
            [ agent.learning_rate, agent.reward_record ] for agent in agents
        ],
        "ALL AGENTS CULMUTATIVE REWARD OVER THE EPOCHS COMPARISION",
        "all_agents_culmutative_reward_over_the_epochs_comparision"
    )

    # Best agent mean rewards record
    Plots.plot_cum_total_rewards_policies(
        EPOCHS,
        [
            [ best_agent.learning_rate, best_agent.list_mean_reward ]
        ],
        "BEST AGENT'S MEAN REWARD OVER THE EPOCHS",
        'best_agent_mean_reward_over_the_epochs'
    )

    # Agents mean reward over the epochs comparision
    Plots.plot_cum_total_rewards_policies(
        EPOCHS,
        [
            [ agent.learning_rate, agent.list_mean_reward ] for agent in agents
        ],
        "ALL AGENTS MEAN REWARD OVER THE EPOCHS COMPARISION",
        'all_agents_mean_reward_over_the_epochs_comparision'
    )

    print('\033[91mLOADING THE PLOTS\033[m')

    print('-'*50)


if __name__ == "__main__":
    main()
