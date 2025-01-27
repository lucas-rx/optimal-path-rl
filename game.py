import numpy as np

from assets import *
from typing import List

REWARD_NORMAL = -0.04
REWARD_VICTORY = 1
REWARD_DEFEAT = -1

CORRECT_ACTION_PROBA = 0.8
INCORRECT_ACTION_PROBA = (1 - CORRECT_ACTION_PROBA) / 2

class Game:
    def __init__(self, grid: np.ndarray):
        self.grid = grid

        self.nb_rows = self.grid.shape[0]
        self.nb_columns = self.grid.shape[1]

    # Etats / coordonnées
        
    def change_state_to_2d_coos(self, state: int) -> List[int]:
        """
        Retourne les coordonnées 2D d'un état.
        Exemple avec une grille de 4x3 : 9 -> [2, 1]
        """
        return [state // self.nb_columns, state % self.nb_columns]
    
    def change_2d_coos_to_state(self, _2d_coos: List[int]) -> int:
        """
        Retourne le numéro d'un état à partir de ses coorodnnées 2D.
        Exemple avec une grille de 4x3 : [2, 1] -> 9
        """
        return _2d_coos[0] * self.nb_columns + _2d_coos[1]

    # Grille d'emojis

    def create_emoji_grid(self) -> np.ndarray:
        """
        Crée une grille représentant l'état initial avec des emojis.
        """

        emoji_grid_dict = {
            0: NORMAL,
            1: VICTORY,
            2: DEFEAT,
            3: WALL
        }

        emoji_grid = []
        for i in range(self.nb_rows):

            emoji_row = []
            for j in range(self.nb_columns):
                emoji_row.append(emoji_grid_dict[self.get_cell_type_from_2d_coos([i, j])])
            emoji_grid.append(emoji_row)

        return emoji_grid

    def update_emoji_grid(self, initial_position: int, next_position: int) -> None:
        """
        Met à jour la grille d'emojis.
        """

        initial_position = self.change_state_to_2d_coos(initial_position)
        next_position = self.change_state_to_2d_coos(next_position)

        self.emoji_grid[initial_position[0]][initial_position[1]] = PATH
        self.emoji_grid[next_position[0]][next_position[1]] = PACMAN
    
    # Type de cellule (0, 1, 2, 3)
    
    def get_cell_type_from_state(self, state: int) -> int:
        """
        Retourne le type de cellule (0, 1, 2 ou 3) à partir du numéro de l'état.
        """
        _2d_coos = self.change_state_to_2d_coos(state)
        return self.grid[_2d_coos[0], _2d_coos[1]]
    
    def get_cell_type_from_2d_coos(self, _2d_coos: List[int]) -> int:
        """
        Retourne le type de cellule (0, 1, 2 ou 3) à partir des coordonnées de l'état.
        """

        return self.grid[_2d_coos[0], _2d_coos[1]]
    
    # Récompense
        
    def get_reward(self, state: int) -> int:
        """
        Retourne la récompense associée à un état.
        """

        cell_type = self.get_cell_type_from_state(state)
        cell_type_dict = {
            0: REWARD_NORMAL,
            1: REWARD_VICTORY,
            2: REWARD_DEFEAT,
            3: 0
        }

        return cell_type_dict[cell_type]

    def get_state_after_action(self, initial_state: int, action: str) -> int:
        """
        Calcule l'état suivant, à partir de l'état précédent et d'une action.
        """

        initial_state_2d_coos = self.change_state_to_2d_coos(initial_state)

        next_state_dict = { 
            "UP": [initial_state_2d_coos[0] - 1, initial_state_2d_coos[1]],
            "DOWN": [initial_state_2d_coos[0] + 1, initial_state_2d_coos[1]],
            "LEFT": [initial_state_2d_coos[0], initial_state_2d_coos[1] - 1],
            "RIGHT": [initial_state_2d_coos[0], initial_state_2d_coos[1] + 1]
        }
        next_state_2d_coos = next_state_dict[action]
        next_state_2d_coos = self.check_coos_bounds(next_state_2d_coos)
        next_state_2d_coos = self.check_walls(initial_state_2d_coos, next_state_2d_coos)

        return self.change_2d_coos_to_state(next_state_2d_coos)
    
    def check_coos_bounds(self, _2d_coos: List[int]) -> List[int]:
        """
        Vérifie la validité des coordonnées, et les modifie si nécessaire.
        """

        if _2d_coos[0] < 0:
            _2d_coos[0] = 0
        if _2d_coos[0] > self.nb_rows - 1:
            _2d_coos[0] = self.nb_rows - 1

        if _2d_coos[1] < 0:
            _2d_coos[1] = 0
        if _2d_coos[1] > self.nb_columns - 1:
            _2d_coos[1] = self.nb_columns - 1

        return _2d_coos
    
    def check_walls(self, initial_state_2d_coos: List[int], next_state_2d_coos: List[int]) -> List[int]:
        """
        Vérifie si l'état suivant est un mur ou non. Si oui, annule la modification des coordonnées.
        """

        next_state_cell_type = self.get_cell_type_from_2d_coos(next_state_2d_coos)
        if next_state_cell_type == 3:
            next_state_2d_coos = initial_state_2d_coos.copy()
        return next_state_2d_coos
    
    def get_orthogonal_states_after_action(self, initial_state: int, action: str) -> List[int]:
        """
        Calcule les états orthogonaux à l'action souhiatée (là où l'on atterrira si l'on glisse).
        """

        orthogonal_actions_dict = {
            "UP": ["LEFT", "RIGHT"],
            "DOWN": ["LEFT", "RIGHT"],
            "LEFT": ["UP", "DOWN"],
            "RIGHT": ["UP", "DOWN"]
        }
        orthogonal_actions = orthogonal_actions_dict[action]

        orthogonal_states = []
        for action_ in orthogonal_actions:
            orthogonal_states.append(self.get_state_after_action(initial_state, action_))

        return orthogonal_states
    
    def compute_probability(self, next_state: int, initial_state: int, action: str) -> float:
        """
        Calcule P(next_state | initial_state, action).
        """

        if self.get_cell_type_from_state(initial_state) == 3:
            return 0

        desired_state = self.get_state_after_action(initial_state, action)
        orthogonal_states = self.get_orthogonal_states_after_action(initial_state, action)

        probability = 0
        if next_state == desired_state:
            probability += CORRECT_ACTION_PROBA
        for state in orthogonal_states:
            if next_state == state:
                probability += INCORRECT_ACTION_PROBA

        return probability
    
    def find_next_action(self, initial_position: int, next_position: int, optimal_action: str) -> str:
        """
        Permet de trouver quelle action a été faite alors que Pac-Man a glissé.
        """
        possible_actions_dict = {
            "UP": ["UP", "LEFT", "RIGHT"],
            "DOWN": ["DOWN", "LEFT", "RIGHT"],
            "LEFT": ["UP", "DOWN", "RIGHT"],
            "RIGHT": ["UP", "DOWN", "LEFT"]
        }
        possible_actions = possible_actions_dict[optimal_action]

        for possible_action in possible_actions:
            if self.get_state_after_action(initial_position, possible_action) == next_position:
                return possible_action
            
