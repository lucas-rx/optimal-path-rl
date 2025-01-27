import numpy as np
import random

from assets import *
from typing import Dict, List

policy_grid_dict = {
    "UP": UP,
    "DOWN": DOWN,
    "LEFT": LEFT,
    "RIGHT": RIGHT,
}

def stringify_grid(grid: np.ndarray) -> str:
    """
    Transforme la grille d'emojis en string.
    """

    buffer = ""
    for row in grid:
        buffer += "".join(row) + "\n"
    return buffer

def max_dict(dict_: Dict[str, int]) -> int:
    """
    Retourne la valeur maximale d'un dictionnaire.
    """

    keys = list(dict_.keys())
    max_value = dict_[keys[0]]

    for _, value in dict_.items():
        if value > max_value:
            max_value = value

    return max_value

def argmax_dict(dict_: Dict[str, int]) -> str:
    """
    Retourne la clé donnant la valeur maximale d'un dictionnaire.
    """

    keys = list(dict_.keys())
    max_key = keys[0]
    max_value = dict_[keys[0]]

    for key, value in dict_.items():
        if value > max_value:
            max_key = key
            max_value = value

    return max_key

def argmax_list(list_: List[int]) -> int:
    """
    Retourne l'indice donnant la valeur maximale de la liste.
    """

    max_index = 0
    max_value = list_[0]

    for index, value in enumerate(list_):
        if value > max_value:
            max_index = index
            max_value = value

    return max_index

def random_argmax_dict(dict_: Dict[str, int]) -> str:

    keys = list(dict_.keys())
    max_key = []
    max_value = dict_[keys[0]]

    for key, value in dict_.items():
        if value > max_value:
            max_key = [key]
            max_value = value
        elif value == max_value:
            max_key.append(key)

    return random.choice(max_key)
