from src.robot import *
import numpy as np
from src.plot_funcs import *

def main():
    np.random.seed(42)

    # ==== Dados para o heatmap (2 níveis de bateria × 3 ações) ====
    heatmap_array = np.random.randint(0, 10, size=(2, 3))

    # ==== Dados para a melhor política ====
    epochs = 50
    n_runs = 5
    rewards_best_policy = [
        np.cumsum(np.random.normal(loc=1.0, scale=0.5, size=epochs))
        for _ in range(n_runs)
    ]
    list_rewards_best_policy = (epochs, rewards_best_policy)

    # ==== Dados para diferentes valores de λ ====
    lambdas = [0.1, 0.3, 0.7, 0.9]
    list_all_rewards_learning = (
        epochs,
        [
            {
                "lam": lam,
                "rewards": [
                    np.cumsum(np.random.normal(loc=1.0 + lam, scale=0.5, size=epochs))
                    for _ in range(n_runs)
                ],
            }
            for lam in lambdas
        ],
    )

    # ==== Criar objeto e gerar plots ====
    plots = Plots(heatmap_array, list_rewards_best_policy, list_all_rewards_learning)

    plots.plot_policy_heatmap()
    plots.plot_cum_total_rewards_best_policy()
    plots.plot_avg_learning_rewards()



if __name__ == "__main__":
    main()
