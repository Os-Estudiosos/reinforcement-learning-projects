import numpy as np
from src.ambient import Ambient


def main():
    ambient = Ambient(
        .5,
        .5,
        .01,
        1000
    )

    ambient.run()

    A = np.exp(ambient.agent.numeric_preferences)
    row_sums = A.sum(axis=1, keepdims=True)  # soma de cada linha
    P = A / row_sums

    print(ambient.agent.numeric_preferences)
    print(A)
    print(P)


if __name__ == "__main__":
    main()
