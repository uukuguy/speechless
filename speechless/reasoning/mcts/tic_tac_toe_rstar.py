from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List
import math, random
import numpy as np
from speechless.reasoning.mcts.mcts_rstar import MCTS_Node, MCTS_Searcher

class TicTacToeNode(MCTS_Node):
    def __init__(self, board, player):
        super().__init__()
        self.board = board
        self.player = player

    def find_children(self, rollout_id):
        children = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    new_board = [row[:] for row in self.board]
                    new_board[i][j] = self.player
                    children.append(TicTacToeNode(new_board, 3 - self.player))
        return children

    def is_terminal(self):
        return self.check_winner() is not None or all(all(cell != 0 for cell in row) for row in self.board)

    def calculate_reward(self):
        winner = self.check_winner()
        if winner is None:
            return 0.5  # Draw
        return 1.0 if winner == 1 else 0.0

    def skip_backprop(self):
        return False

    def check_winner(self):
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != 0:
                return self.board[i][0]
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != 0:
                return self.board[0][i]
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != 0:
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != 0:
            return self.board[0][2]
        return None

def play_game():
    searcher = MCTS_Searcher(
        exploration_weight=1.0,
        weight_scheduler="const",
        num_rollouts=1000,
        discount=1.0,
        verbose=False
    )

    board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    root = TicTacToeNode(board, 1)

    while not root.is_terminal():
        for _ in range(searcher.num_rollouts):
            searcher.do_rollout(root, _)

        best_child = max(searcher.parent2children[root], key=lambda n: searcher.N[n])
        root = best_child
        print_board(root.board)
        print()

    winner = root.check_winner()
    if winner:
        print(f"Player {winner} wins!")
    else:
        print("It's a draw!")

def print_board(board):
    for row in board:
        print(" ".join("X" if cell == 1 else "O" if cell == 2 else "-" for cell in row))

if __name__ == "__main__":
    play_game()