import numpy as np
import pandas as pd
import time

from copy import deepcopy
from game import Game
from utils import *

class QLearningAlgorithm(Game):

    def __init__(self):
        self.game = super().__init__(pd.read_csv("Q-Learning.txt", header=None).to_numpy())

        self.gamma = self.grid[self.nb_rows - 3, 0]
        self.alpha = self.grid[self.nb_rows - 2, 0]
        self.nb_essais = int(self.grid[self.nb_rows - 1, 0])

        self.grid = self.grid[:self.nb_rows - 3, :]
        self.nb_rows -= 3

        self.emoji_grid = self.create_emoji_grid()

        self.initial_position = self.change_2d_coos_to_state([self.nb_rows - 1, 0])
        self.Q_table = self.create_Q_table()
        self.frequencies_state_action_table = np.zeros((self.nb_rows * self.nb_columns, 4))

        self.buffer = f"""----- ALGORITHME Q-LEARNING -----

Grille ({self.nb_rows} x {self.nb_columns}) :
{stringify_grid(self.emoji_grid)}
Gamma : {self.gamma}
Alpha : {self.alpha}
Nombre d'essais : {self.nb_essais}
"""

    def create_Q_table(self) -> None:
        """
        Crée une Q-table.
        """

        actions_dict = {
            "UP": 0,
            "DOWN": 0,
            "LEFT": 0,
            "RIGHT": 0
        }

        Q_table = []
        for _ in range(self.nb_rows * self.nb_columns):
            Q_table.append(deepcopy(actions_dict))
        return Q_table
    
    def display_Q_table(self) -> None:
        """
        Affiche joliment la Q-table.
        """
        buffer = ""
        for state in range(self.nb_rows * self.nb_columns):
            buffer += f"Etat {state} : {self.Q_table[state]}"
            if state != self.nb_rows * self.nb_columns - 1:
                buffer += "\n"
        return buffer

    def compute_next_Q_value(self) -> str:
        """
        Calcule la Q-value suivante, en choisissant l'action avec le Q(s, a) maximal.
        """

        actions_initial_position = self.Q_table[self.initial_position]
        optimal_action = random_argmax_dict(actions_initial_position)

        next_position = self.get_state_after_action(self.initial_position, optimal_action)
        actions_next_position = self.Q_table[next_position]

        self.Q_table[self.initial_position][optimal_action] = self.Q_table[self.initial_position][optimal_action] + self.alpha * (self.get_reward(next_position) + self.gamma * max_dict(actions_next_position) - self.Q_table[self.initial_position][optimal_action])
        self.initial_position = next_position

    def train(self) -> None:

        start = time.time()
        for i in range(self.nb_essais):
            if i % 100 == 0:
                print(i)

            self.buffer += f"\n----- ITERATION {i + 1} / {self.nb_essais} -----\n"
            self.initial_position = self.change_2d_coos_to_state([self.nb_rows - 1, 0])

            while True:

                self.compute_next_Q_value()
                if self.get_cell_type_from_state(self.initial_position) == 1:
                    self.buffer += f"""
Victoire ! Q_table =
{self.display_Q_table()}
"""
                    break
                elif self.get_cell_type_from_state(self.initial_position) == 2:
                    self.buffer += f"""
Défaite ! Q_table =
{self.display_Q_table()}
"""
                    break

        stop = int(time.time() - start)
        self.buffer += f"\nTemps d'exécution : {str(stop // 60).rjust(2, '0')} : {str(stop % 60).rjust(2, '0')}\n"
        self.compute_optimal_policy()

    def compute_optimal_policy(self) -> None:
        """
        Calcule la politique optimale.
        """

        self.policy = {}
        self.buffer += "\n----- POLITIQUE OPTIMALE -----\n\n"

        for state in range(self.nb_rows * self.nb_columns):

            if self.get_cell_type_from_state(state) == 0:

                Q_values = self.Q_table[state]
                optimal_action = argmax_dict(Q_values)
                self.policy[state] = optimal_action

                self.buffer += f"État {state} : action {optimal_action} ({Q_values})\n"

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
{stringify_grid(policy_grid)}"""

    def write_report(self) -> None:
        """
        Ecrit le rapport dans un fichier de log.
        """

        with open("log-file_QL.txt", "w") as f:
            f.write(self.buffer)

if __name__ == "__main__":
    QLA = QLearningAlgorithm()
    QLA.train()
    QLA.write_report()
