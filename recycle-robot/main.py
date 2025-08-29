from src.ambient import Ambient


def main():
    ambient = Ambient(
        .5,
        .5,
        .01,
        1000
    )

    ambient.run()

    print(ambient.agent.numeric_preferences)


if __name__ == "__main__":
    main()
