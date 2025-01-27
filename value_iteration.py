import numpy as np
import pandas as pd
import random
import time

from assets import *
from copy import deepcopy
from game import Game
from typing import Dict
from utils import *

class ValueIterationAlgorithm(Game):

    def __init__(self):
        self.game = super().__init__(pd.read_csv("value-iteration.txt", header=None).to_numpy())

        self.gamma = self.grid[self.nb_rows - 2, 0]
        self.tolerance = self.grid[self.nb_rows - 1, 0]

        self.grid = self.grid[:self.nb_rows - 2, :]
        self.nb_rows -= 2

        self.emoji_grid = self.create_emoji_grid()

        self.U = self.create_U()
        self.next_U = np.zeros(self.nb_rows * self.nb_columns)

        self.buffer = f"""----- ALGORITHME VALUE ITERATION -----

Grille ({self.nb_rows} x {self.nb_columns}) :
{stringify_grid(self.emoji_grid)}
Gamma : {self.gamma}
Tolérance : {self.tolerance}

U =
{self.U.reshape((self.nb_rows, self.nb_columns))}

"""
    
    def create_U(self) -> np.ndarray:
        """
        Initialise U avec les valeurs des récompenses des états.
        """

        U = np.zeros(self.nb_rows * self.nb_columns)
        for i in range(self.nb_rows):
            for j in range(self.nb_columns):
                state = self.change_2d_coos_to_state([i, j])
                U[state] = self.get_reward(state)
        return U
    
    def replace_next_U_values_for_terminal_states(self) -> None:
        """
        Remplace les valeurs des états terminaux de U' par leur récompense.
        """

        for i in range(self.nb_rows):
            for j in range(self.nb_columns):
                if self.grid[i][j] == 1 or self.grid[i][j] == 2:
                    state = self.change_2d_coos_to_state([i, j])
                    self.next_U[state] = self.get_reward(state)

    def compute_Q_value(self, initial_state: int, action: str) -> float:
        """
        Calcule Q(s, a).
        """

        Q_value = 0
        for next_state in range(self.nb_rows * self.nb_columns):
            Q_value += self.compute_probability(next_state, initial_state, action) * (self.get_reward(next_state) + self.gamma * self.U[next_state])
            # self.buffer += f"({next_state} | {initial_state}, {action}) : {self.compute_probability(next_state, initial_state, action)} * {self.get_reward(next_state)} + {self.U[next_state]} = {Q_value}\n"
        return Q_value
    
    def compute_Q_value_for_state(self, state: int) -> Dict[str, float]:
        """
        Calcule Q(s, a) pour toute action a étant donné s.
        """

        Q_value_results = {
            "UP": 0,
            "DOWN": 0,
            "LEFT": 0,
            "RIGHT": 0
        }
        
        for action in ["UP", "DOWN", "LEFT", "RIGHT"]:
            Q_value_results[action] = self.compute_Q_value(state, action)
        return Q_value_results
    
    def train(self) -> None:
        """
        Entraîne l'algorithme (itère sur U).
        """

        start = time.time()
        current_iteration = 1

        while True:

            for state in range(self.nb_rows * self.nb_columns):

                Q_value_results = self.compute_Q_value_for_state(state)
                self.next_U[state] = max_dict(Q_value_results)

            self.replace_next_U_values_for_terminal_states()
            delta_U = round(np.sum(np.abs(self.U - self.next_U)), 8)

            self.buffer += f"""----- ITERATION {current_iteration} -----

U{current_iteration} =
{self.next_U.reshape((self.nb_rows, self.nb_columns))}
Delta = {delta_U} (tol. {self.tolerance})
            
"""
            # print(f"It. = {current_iteration}, delta = {delta_U} (tol. {self.tolerance})")

            if delta_U <= self.tolerance:
                break

            current_iteration += 1
            self.U = self.next_U.copy()

        stop = int(time.time() - start)
        self.buffer += f"""Algorithme terminé en {current_iteration} itérations
Temps d'exécution : {str(stop // 60).rjust(2, "0")} : {str(stop % 60).rjust(2, "0")}\n
"""

        self.compute_optimal_policy()

    def compute_optimal_policy(self) -> None:
        """
        Calcule la politique optimale.
        """

        self.policy = {}
        self.buffer += "----- POLITIQUE OPTIMALE -----\n\n"

        for state in range(self.nb_rows * self.nb_columns):

            if self.get_cell_type_from_state(state) == 0:

                Q_value_results = self.compute_Q_value_for_state(state)
                optimal_action = argmax_dict(Q_value_results)
                self.policy[state] = optimal_action
                
                self.buffer += f"État {state} : action {optimal_action} ({Q_value_results})\n"

        self.create_policy_grid()

    def create_policy_grid(self) -> None:
        """
        Crée une grille affichant, pour chaque état, l'action optimale.
        """

        policy_grid = deepcopy(self.emoji_grid)
        for state in self.policy.keys():
            _2d_coos = self.change_state_to_2d_coos(state)
            policy_grid[_2d_coos[0]][_2d_coos[1]] = policy_grid_dict[self.policy[state]]

        self.buffer += f"""
Grille des actions optimales :
{stringify_grid(policy_grid)}
"""

    def simulate_pacman(self) -> None:
        """
        Simule une partie du jeu. Les actions sont tirées selon la distribution de probabilité.
        """

        initial_position = self.change_2d_coos_to_state([self.nb_rows - 1, 0])
        self.emoji_grid[self.nb_rows - 1][0] = PACMAN
        self.buffer += f"""----- SIMULATION DE L'ALGORITHME -----

Position de départ (état {initial_position}) :
{stringify_grid(self.emoji_grid)}
"""

        while self.get_cell_type_from_state(initial_position) == 0:

            optimal_action = self.policy[initial_position]

            probabilities = []
            weights = []
            for next_state in range(self.nb_rows * self.nb_columns):
                probability = self.compute_probability(next_state, initial_position, optimal_action)
                if probability != 0:
                    probabilities.append(next_state)
                    weights.append(probability)

            optimal_position = probabilities[argmax_list(weights)]
            next_position = random.choices(probabilities, weights=weights)[0]
            
            self.update_emoji_grid(initial_position, next_position)

            if optimal_position == next_position:
                self.buffer += f"Action {optimal_action} {policy_grid_dict[optimal_action]}\n\n"
            else:
                next_action = self.find_next_action(initial_position, next_position, optimal_action)
                self.buffer += f"Glissade : Action {next_action} {policy_grid_dict[next_action]} au lieu de {optimal_action} {policy_grid_dict[optimal_action]} \n\n"
            
            self.buffer += f"Position actuelle (état {next_position}) :\n{stringify_grid(self.emoji_grid)}\n"
            
            if self.get_cell_type_from_state(next_position) == 1:
                self.buffer += "Victoire !\n"
                break
            elif self.get_cell_type_from_state(next_position) == 2:
                self.buffer += "Défaite !\n"
                break
            
            initial_position = next_position

    def write_report(self) -> None:
        """
        Ecrit le rapport dans un fichier de log.
        """

        with open("log-file_VI.txt", "w") as f:
            f.write(self.buffer)

if __name__ == "__main__":
    VIA = ValueIterationAlgorithm()
    VIA.train()
    VIA.simulate_pacman()
    VIA.write_report()

