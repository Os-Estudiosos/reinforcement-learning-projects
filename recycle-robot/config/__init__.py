import os


def FIGURES_FOLDER() -> str:
    """Return the absolute path of the datasets

    Returns:
        str: Dataset's path
    """
    return os.path.join(os.getcwd(), 'figures')

def clear_terminal():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')