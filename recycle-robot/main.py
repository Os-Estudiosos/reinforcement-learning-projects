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


if __name__ == "__main__":
    main()
