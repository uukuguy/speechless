import numpy as np
from typing import List, Tuple, Optional
from speechless.reasoning.mcts.mcts_reasoner import MCTS, MCTSNode, WorldModel, SearchConfig, State, Action, Example

# 定义游戏状态
class TicTacToeState:
    def __init__(self, board: np.ndarray, player: int):
        self.board = board
        self.player = player

    def __str__(self):
        return f"Board:\n{self.board}\nPlayer: {self.player}"

# 定义动作
class TicTacToeAction:
    def __init__(self, row: int, col: int):
        self.row = row
        self.col = col

    def __str__(self):
        return f"Place at ({self.row}, {self.col})"

# 实现世界模型
class TicTacToeWorldModel(WorldModel[TicTacToeState, TicTacToeAction, None]):
    def init_state(self) -> TicTacToeState:
        return TicTacToeState(np.zeros((3, 3), dtype=int), 1)

    def step(self, state: TicTacToeState, action: TicTacToeAction) -> Tuple[TicTacToeState, dict]:
        new_board = state.board.copy()
        new_board[action.row, action.col] = state.player
        new_state = TicTacToeState(new_board, 3 - state.player)  # Switch player
        return new_state, {}

    def is_terminal(self, state: TicTacToeState) -> bool:
        # Check rows, columns, and diagonals for a win
        for i in range(3):
            if abs(np.sum(state.board[i, :])) == 3 or abs(np.sum(state.board[:, i])) == 3:
                return True
        if abs(np.trace(state.board)) == 3 or abs(np.trace(np.fliplr(state.board))) == 3:
            return True
        # Check for a draw
        if np.all(state.board != 0):
            return True
        return False

# 实现搜索配置
class TicTacToeSearchConfig(SearchConfig[TicTacToeState, TicTacToeAction, None]):
    def get_actions(self, state: TicTacToeState) -> List[TicTacToeAction]:
        actions = []
        for i in range(3):
            for j in range(3):
                if state.board[i, j] == 0:
                    actions.append(TicTacToeAction(i, j))
        return actions

    def fast_reward(self, state: TicTacToeState, action: TicTacToeAction) -> Tuple[float, dict]:
        return 0, {}  # 简单起见，我们不使用启发式奖励

    def reward(self, state: TicTacToeState, action: TicTacToeAction, **kwargs) -> Tuple[float, dict]:
        new_state, _ = TicTacToeWorldModel().step(state, action)
        if TicTacToeWorldModel().is_terminal(new_state):
            # 检查是否获胜
            for i in range(3):
                if abs(np.sum(new_state.board[i, :])) == 3 or abs(np.sum(new_state.board[:, i])) == 3:
                    return 1 if state.player == 1 else -1, {}
            if abs(np.trace(new_state.board)) == 3 or abs(np.trace(np.fliplr(new_state.board))) == 3:
                return 1 if state.player == 1 else -1, {}
            # 如果是平局
            return 0, {}
        return 0, {}

# 主函数
def main():
    world_model = TicTacToeWorldModel()
    search_config = TicTacToeSearchConfig()
    
    mcts = MCTS(n_iters=1000, depth_limit=9, output_strategy='max_reward')
    
    state = world_model.init_state()
    print("Initial state:")
    print(state)
    
    while not world_model.is_terminal(state):
        result = mcts(world_model, search_config)
        if result.trace is None or len(result.trace[1]) == 0:
            print("No valid moves found. Game over.")
            break
        
        best_action = result.trace[1][0]
        print(f"\nBest action: {best_action}")
        state, _ = world_model.step(state, best_action)
        print("\nNew state:")
        print(state)
    
    print("\nGame over!")

if __name__ == "__main__":
    main()