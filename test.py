import pickle
import numpy as np

import moves
from logic import UP, DOWN, LEFT, RIGHT, Board


if __name__ == '__main__':
    left_moves, right_moves = moves.load_moves()

    board = Board(left_moves, right_moves)
    test1 = np.array([
        [0, 1, 0, 1],
        [8, 0, 8, 8],
        [1, 1, 1, 1],
        [1, 1, 2, 2]
    ])

    board.tiles = test1.copy()
    board.swipe(UP)
    print(test1)
    print("UP")
    print(board.tiles)
    print(board.move_exists())
    print()

    board.tiles = test1.copy()
    board.swipe(DOWN)
    print(test1)
    print("DOWN")
    print(board.tiles)
    print(board.move_exists())
    print()

    board.tiles = test1.copy()
    board.swipe(LEFT)
    print(test1)
    print("LEFT")
    print(board.tiles)
    print(board.move_exists())
    print()

    board.tiles = test1.copy()
    board.swipe(RIGHT)
    print(test1)
    print("RIGHT")
    print(board.tiles)
    print(board.move_exists())

    board.tiles = np.array([
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [1, 2, 3, 4],
        [4, 3, 2, 1]
    ])
    print(board.tiles)
    print(board.move_exists())